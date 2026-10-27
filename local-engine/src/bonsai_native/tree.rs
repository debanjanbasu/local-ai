//! `KFlash` tree verify: the target checks a best-first tree of up to eight
//! draft rows in one pass, and commits the longest path it agrees with.
//!
//! Row `r` of a [`DraftTree`] sits at position `start + depth[r]` and is
//! computed as plain decoding of its own path would compute it:
//!
//! - **Attention.** Its K/V row is written at cache row `start + r` with the
//!   rotary position `start + depth[r]`; it attends the history below
//!   `start` through the decode split kernel of that prefix (or, from three
//!   rows on a Q8 cache, the verify block's row kernel with every row's
//!   prefix pinned at `start`), and its ancestors and itself among the
//!   pass's rows key by key in path order
//!   ([`local_metal::bonsai_ops::BonsaiOps::attention_tree_kv`]).
//! - **Gated `DeltaNet`.** Its short convolution reads its ancestors' inputs,
//!   and its recurrence starts from its parent's state: rebuilt from the
//!   committed state by replaying its ancestors' updates when the parent is
//!   not the row before it (`bo_conv_l2_decay_tree`, `bo_gdn_tree_rows_4*`).
//!   The committed state and history are only read, as by a chain verify.
//!
//! So a row's numbers depend on `start`, its path and the verify's row count
//! (which picks the projection kernels: one row, two to four, five to
//! eight), never on where its ancestors sit or on the other rows. A chain
//! (`parent[r] = r - 1`) is verified by the existing chain verify block, and
//! a one-row tree by a plain decode step, so both keep their numbers exactly.
//!
//! The commit moves the accepted path's K/V rows and its recurrence inputs
//! (which the pass recorded on the sequence's GDN tape) to consecutive rows
//! and drops every other row; the next step folds the path's recurrence
//! inputs into the untouched committed state (see `tape`).

use std::time::Duration;

use local_metal::bonsai_ops::{TREE_DESCRIPTOR_WORDS, TREE_MASKS, TREE_PATH, TREE_ROWS};

use super::block::SUBMIT_BEFORE_LAYERS;
use super::layers::decay_projection;
use super::{
    AttentionLayer, BlockOutput, BonsaiModel, CommandBatch, Described, FullAttentionLayer,
    HadamardDirection, MetalBuffer, RecurrentLayer, Rope, SamplingResult, Selection, VOCAB,
    Verifier, WIDTH, decode_embeddings,
};
use crate::sampler::{Sampler, verify_tree_drafts};

/// Rows of the largest tree a verify pass checks.
pub const MAX_TREE_ROWS: usize = TREE_ROWS as usize;

/// A draft tree of up to eight rows.
///
/// Row `r` holds `tokens[r]`, follows row `parents[r]` and sits `depths[r]`
/// positions after the anchor. Row 0 is the anchor (the next token, which
/// the target has not run yet), with parent -1 and depth 0; every parent
/// comes before its children.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
    pub depths: Vec<u32>,
}

impl DraftTree {
    /// The chain `tokens[0]`, `tokens[1]`, ...: parent `r - 1`, depth `r`.
    #[must_use]
    pub fn chain(tokens: &[u32]) -> Self {
        Self {
            tokens: tokens.to_vec(),
            parents: (0..tokens.len() as u32)
                .map(|row| row.cast_signed() - 1)
                .collect(),
            depths: (0..tokens.len() as u32).collect(),
        }
    }

    /// A tree from `tokens` and `parents`, with each row's depth derived.
    pub fn from_parents(tokens: &[u32], parents: &[i32]) -> crate::Result<Self> {
        let mut depths = Vec::with_capacity(parents.len());
        for (row, &parent) in parents.iter().enumerate() {
            let depth = match usize::try_from(parent) {
                Ok(parent) if parent < row => depths[parent] + 1,
                _ if row == 0 && parent == -1 => 0,
                _ => return invalid_tree(),
            };
            depths.push(depth);
        }
        let tree = Self {
            tokens: tokens.to_vec(),
            parents: parents.to_vec(),
            depths,
        };
        tree.validate()?;
        Ok(tree)
    }

    /// Rows in the tree.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tokens.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Whether every row follows the row before it.
    #[must_use]
    pub fn is_chain(&self) -> bool {
        self.parents
            .iter()
            .enumerate()
            .skip(1)
            .all(|(row, &parent)| parent == (row as u32).cast_signed() - 1)
    }

    /// 1..=8 rows, one parent and depth per row, row 0 the root, every
    /// parent before its child and every depth its parent's plus one.
    pub fn validate(&self) -> crate::Result<()> {
        let rows = self.len();
        if !(1..=MAX_TREE_ROWS).contains(&rows)
            || self.parents.len() != rows
            || self.depths.len() != rows
            || self.parents[0] != -1
            || self.depths[0] != 0
            || self.tokens.iter().any(|&token| token as usize >= VOCAB)
        {
            return invalid_tree();
        }
        for row in 1..rows {
            match usize::try_from(self.parents[row]) {
                Ok(parent) if parent < row && self.depths[row] == self.depths[parent] + 1 => {}
                _ => return invalid_tree(),
            }
        }
        Ok(())
    }

    /// Each row's ancestor-or-self mask (bit `j`: row `j`).
    fn masks(&self) -> Vec<u32> {
        let mut masks: Vec<u32> = Vec::with_capacity(self.len());
        for (row, &parent) in self.parents.iter().enumerate() {
            let parent = usize::try_from(parent).map_or(0, |parent| masks[parent]);
            masks.push(parent | 1 << row);
        }
        masks
    }

    /// Whether `path` starts at the root and follows parent links.
    fn is_path(&self, path: &[usize]) -> bool {
        path.first() == Some(&0)
            && path.windows(2).all(|pair| {
                pair[1] < self.len() && usize::try_from(self.parents[pair[1]]) == Ok(pair[0])
            })
    }
}

fn invalid_tree<T>() -> crate::Result<T> {
    Err(crate::Error::InvalidArgument(
        "a draft tree holds 1..=8 rows, row 0 the root, every parent before its child".into(),
    ))
}

/// A row whose parent was accepted, and whether the target took it: the
/// records a drafter's calibration learns from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TestedNode {
    pub row: usize,
    pub depth: u32,
    pub accepted: bool,
}

/// What the acceptance walk of a verified tree found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeAcceptance {
    /// The target's own selection after each accepted row: the accepted
    /// drafts' tokens, then the bonus token, which is the next anchor.
    pub samples: Vec<SamplingResult>,
    /// The accepted rows, root first: the rows [`BonsaiModel::tree_commit`]
    /// commits.
    pub path: Vec<usize>,
    /// Every row whose parent was accepted.
    pub tested: Vec<TestedNode>,
}

/// How the pending verify ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// One row: a plain decode step, already committed.
    Decode,
    /// The chain verify block.
    Chain,
    /// The tree kernels.
    Tree,
}

struct Pending {
    start: usize,
    tree: DraftTree,
    pass: Pass,
}

/// The target's tree verify buffers, made by [`BonsaiModel::enable_tree_verify`].
pub(super) struct TreeVerifier {
    /// Eight rows of recurrence inputs per GDN layer, logits and, for a
    /// chain, the copy-free rollback's final state and history.
    pub(super) verifier: Verifier,
    /// Parents, ancestor masks and a commit's path (`TREE_*` words).
    descriptor: MetalBuffer,
    /// Tree attention's partial records, grown with the anchor position.
    partials: MetalBuffer,
    pending: Option<Pending>,
}

/// The tree whose first `n` rows [`BonsaiModel::tree_verify_costs`] times: a
/// chain at two rows, branching from three on.
const COST_PARENTS: [i32; MAX_TREE_ROWS] = [-1, 0, 0, 1, 1, 3, 2, 5];

impl BonsaiModel {
    /// Make the tree verify buffers: an eight-row verifier (about 100 MB of
    /// recurrence inputs, rollback state and logits), the descriptor and the
    /// attention partials, and room for eight rows of argmax and activations.
    pub fn enable_tree_verify(&mut self) -> crate::Result<()> {
        if self.tree.is_some() {
            return Ok(());
        }
        let shaders = local_metal::shaders::ShaderLibrary::new(self.context.device())?;
        if self.greedy.max_rows() < MAX_TREE_ROWS {
            self.greedy =
                local_metal::draft::GreedyRows::new(&self.context, &shaders, VOCAB, MAX_TREE_ROWS)?;
        }
        if self.block_rows < MAX_TREE_ROWS {
            self.scratch = super::Scratch::new(&self.context, self.kv_allocated, MAX_TREE_ROWS)?;
            self.block_rows = MAX_TREE_ROWS;
        }
        let verifier = Verifier::new(
            &self.context,
            self.state_format,
            MAX_TREE_ROWS - 1,
            MAX_TREE_ROWS - 1,
        )?;
        let partials = MetalBuffer::empty(
            self.context.device(),
            self.ops
                .tree_partial_bytes(self.kv_layout, self.position as u32, TREE_ROWS),
        )?;
        self.info.state_bytes += Verifier::checkpoint_bytes(MAX_TREE_ROWS - 1)
            + MAX_TREE_ROWS * VOCAB * size_of::<f32>();
        self.tree = Some(TreeVerifier {
            verifier,
            descriptor: MetalBuffer::empty(
                self.context.device(),
                TREE_DESCRIPTOR_WORDS * size_of::<i32>(),
            )?,
            partials,
            pending: None,
        });
        Ok(())
    }

    /// Verify `tree` at the current position in one pass, leaving every
    /// row's target logits ([`Self::tree_logits`]), the argmax of each on the
    /// GPU ([`Self::tree_argmax`]) and, for samplers that need them, each
    /// row's best logits and log-probability statistics, until
    /// [`Self::tree_accept`] and [`Self::tree_commit`]. The committed
    /// position does not move, except that a one-row tree is a plain decode
    /// step and commits its row at once.
    pub fn tree_verify(&mut self, tree: &DraftTree) -> crate::Result<()> {
        tree.validate()?;
        let Some(mut state) = self.tree.take() else {
            return Err(crate::Error::InvalidArgument(
                "tree verify is not enabled".into(),
            ));
        };
        state.pending = None;
        let start = self.position;
        let result = if tree.len() == 1 {
            self.forward(tree.tokens[0], true).map(|()| Pass::Decode)
        } else if tree.is_chain() {
            let result = self.forward_block(&tree.tokens, BlockOutput::Verify(&state.verifier));
            self.position = start;
            result.map(|()| Pass::Chain)
        } else {
            self.encode_tree(&mut state, tree).map(|()| Pass::Tree)
        };
        if let Ok(pass) = result {
            state.pending = Some(Pending {
                start,
                tree: tree.clone(),
                pass,
            });
        }
        self.tree = Some(state);
        result.map(|_| ())
    }

    /// Row `row`'s target logits from the pending tree verify.
    #[expect(unsafe_code)]
    #[must_use]
    pub fn tree_logits(&self, row: usize) -> Option<Vec<f32>> {
        let state = self.tree.as_ref()?;
        let pending = state.pending.as_ref()?;
        if row >= pending.tree.len() {
            return None;
        }
        // SAFETY: every block waits for its GPU work before returning, and
        // the borrow ends with this copy.
        Some(unsafe {
            match pending.pass {
                Pass::Decode => self.scratch.logits.as_slice::<f32>()[..VOCAB].to_vec(),
                Pass::Chain | Pass::Tree => state.verifier.verify_logits.as_slice::<f32>()
                    [row * VOCAB..(row + 1) * VOCAB]
                    .to_vec(),
            }
        })
    }

    /// Each row's argmax (`f32::total_cmp` order, ties to the lower id),
    /// selected on the GPU by the pending tree verify.
    #[must_use]
    pub fn tree_argmax(&self) -> Option<Vec<u32>> {
        let pending = self.tree.as_ref()?.pending.as_ref()?;
        Some(
            self.greedy
                .results(pending.tree.len())
                .iter()
                .map(|result| result.best_id)
                .collect(),
        )
    }

    /// The acceptance walk of the pending tree verify: from the root, take
    /// the target's own selection at each row (the GPU argmax for a greedy
    /// sampler, else a draw with the random number plain decoding draws at
    /// that position, which siblings therefore share) and step into the
    /// child holding it, for as long as one does.
    pub fn tree_accept(&mut self, sampler: &mut Sampler) -> crate::Result<TreeAcceptance> {
        let Some(mut state) = self.tree.take() else {
            return Err(crate::Error::InvalidArgument(
                "tree verify is not enabled".into(),
            ));
        };
        let result = self.walk(&mut state, sampler);
        self.tree = Some(state);
        result
    }

    fn walk(
        &mut self,
        state: &mut TreeVerifier,
        sampler: &mut Sampler,
    ) -> crate::Result<TreeAcceptance> {
        let Some(pending) = state.pending.as_ref() else {
            return Err(crate::Error::InvalidArgument(
                "no tree verify is pending".into(),
            ));
        };
        let tree = pending.tree.clone();
        let rows = tree.len();
        let verified = if pending.pass == Pass::Decode {
            verify_tree_drafts(sampler, &tree.tokens, &tree.parents, |_, sampler| {
                self.sample(sampler)
            })?
        } else {
            let selected =
                (self.selection == Selection::Rows(rows)).then(|| self.greedy.results(rows));
            let logits = &mut state.verifier.verify_logits;
            verify_tree_drafts(sampler, &tree.tokens, &tree.parents, |row, sampler| {
                let distribution = sampler
                    .logprobs()
                    .and_then(|top| self.distributions.row(Described::Rows(rows), row, top));
                let top_k = self.candidates.row(Described::Rows(rows), row);
                sampler.select_row(
                    logits,
                    row * VOCAB * size_of::<f32>(),
                    &self.context,
                    &self.sampling,
                    selected.as_ref().map(|results| results[row].best_id),
                    top_k.as_deref(),
                    distribution,
                )
            })?
        };
        Ok(TreeAcceptance {
            samples: verified.samples,
            path: verified.path,
            tested: verified
                .tested
                .into_iter()
                .map(|(row, accepted)| TestedNode {
                    row,
                    depth: tree.depths[row],
                    accepted,
                })
                .collect(),
        })
    }

    /// Commit the pending tree verify's rows `path` (root first, each the
    /// previous one's child): their K/V rows move to consecutive positions,
    /// their recurrence and convolution updates fold into the committed
    /// state, their taps reach the tap ring and the MTP head (if any) ingests
    /// them; every other row is dropped.
    pub fn tree_commit(&mut self, path: &[usize]) -> crate::Result<()> {
        let Some(mut state) = self.tree.take() else {
            return Err(crate::Error::InvalidArgument(
                "tree verify is not enabled".into(),
            ));
        };
        let mut speculation = self.speculation.take();
        let result = self.commit_path(&mut state, speculation.as_mut(), path);
        self.speculation = speculation;
        self.tree = Some(state);
        result
    }

    fn commit_path(
        &mut self,
        state: &mut TreeVerifier,
        head: Option<&mut super::Speculation>,
        path: &[usize],
    ) -> crate::Result<()> {
        let Some(pending) = state.pending.take() else {
            return Err(crate::Error::InvalidArgument(
                "no tree verify is pending".into(),
            ));
        };
        if !pending.tree.is_path(path) {
            return Err(crate::Error::InvalidArgument(
                "a tree commit takes a root-first path of the verified tree".into(),
            ));
        }
        let (start, rows, committed) = (pending.start, pending.tree.len(), path.len());
        let mut batch = CommandBatch::new(&self.context)?;
        match pending.pass {
            // The decode step committed its row already.
            Pass::Decode => {}
            Pass::Chain => {
                self.position = start + rows;
                self.encode_commit_verified(&mut batch, &mut state.verifier, rows, committed)?;
            }
            Pass::Tree => {
                if path.iter().enumerate().any(|(index, &row)| index != row) {
                    self.encode_gather_path(&mut batch, state, start, path)?;
                }
                // The next step folds the path's rows in.
                self.tape.keep(committed);
                self.encode_tap_commit(&mut batch, path, start)?;
            }
        }
        self.position = start + committed;
        if pending.pass == Pass::Decode && head.is_none() {
            return Ok(());
        }
        // Whatever drafted the tree, the head ingests every committed row
        // from the target's own output-normalized hidden, so its cache stays
        // what prefill blocks would leave.
        if let Some(head) = head {
            let tokens = path
                .iter()
                .map(|&row| pending.tree.tokens[row])
                .collect::<Vec<_>>();
            decode_embeddings(
                &self.package,
                &tokens,
                head.mtp.embedding_rows(committed)?,
                None,
            )?;
            self.encode_head_rows(&mut batch, head, committed, start)?;
        }
        self.finish(batch)
    }

    /// Move the accepted rows `path` of a tree verify anchored at `start` to
    /// rows `0..path.len()`: every full-attention layer's K/V cache rows, every
    /// recurrent layer's recurrence inputs and the output-normalized hidden
    /// the head ingests.
    fn encode_gather_path(
        &self,
        batch: &mut CommandBatch,
        state: &TreeVerifier,
        start: usize,
        path: &[usize],
    ) -> crate::Result<()> {
        let mut words = [0i32; TREE_DESCRIPTOR_WORDS];
        for (word, &row) in words[TREE_PATH..].iter_mut().zip(path) {
            *word = (row as u32).cast_signed();
        }
        state
            .descriptor
            .copy_from_bytes(bytemuck::cast_slice(&words), 0);
        let path_words = (&state.descriptor, TREE_PATH * size_of::<i32>());
        let count = path.len();
        let (key_bytes, value_bytes) = (
            self.kv_layout.key.token_bytes(),
            self.kv_layout.value.token_bytes(),
        );
        let mut recurrent = 0;
        for layer in &self.layers {
            match &layer.attention {
                AttentionLayer::Full(full) => self.ops.gather_rows(
                    batch,
                    &[
                        (&full.key_cache, start * key_bytes, key_bytes),
                        (&full.value_cache, start * value_bytes, value_bytes),
                    ],
                    path_words,
                    count,
                )?,
                AttentionLayer::Recurrent(_) => {
                    // The rows the tree recorded on the tape.
                    let step = self.tape.step(recurrent);
                    recurrent += 1;
                    self.ops.gather_rows(
                        batch,
                        &[
                            (&step.raw, 0, 10_240 * size_of::<f32>()),
                            (&step.decay, 0, 48 * size_of::<f32>()),
                            (&step.beta, 0, 48 * size_of::<f32>()),
                        ],
                        path_words,
                        count,
                    )?;
                }
            }
        }
        self.ops.gather_rows(
            batch,
            &[(&self.scratch.normalized, 0, WIDTH * size_of::<f32>())],
            path_words,
            count,
        )?;
        Ok(())
    }

    /// Verify, accept and commit `tree` in one call: what a decoding loop
    /// with a tree drafter runs each cycle.
    pub fn tree_step(
        &mut self,
        tree: &DraftTree,
        sampler: &mut Sampler,
    ) -> crate::Result<TreeAcceptance> {
        self.tree_verify(tree)?;
        let acceptance = self.tree_accept(sampler)?;
        self.tree_commit(&acceptance.path)?;
        Ok(acceptance)
    }

    /// Measured GPU time of a tree verify of 1 to 8 rows at the current
    /// position (index `n - 1`), the median of `repetitions` passes each: a
    /// plain decode step at one row, the chain verify at two, the tree
    /// kernels from three on. Nothing is committed; the model is left as it
    /// was (a pending tree verify is dropped).
    pub fn tree_verify_costs(
        &mut self,
        anchor: u32,
        repetitions: usize,
    ) -> crate::Result<[Duration; MAX_TREE_ROWS]> {
        let repetitions = repetitions.max(1);
        let start = self.position;
        let mut costs = [Duration::ZERO; MAX_TREE_ROWS];
        let median = |mut times: Vec<Duration>| {
            times.sort_unstable();
            times[times.len() / 2]
        };
        // A decode step commits, so restore its state after each one.
        let checkpoint = self.pinned_prompt_checkpoint()?;
        let span = self.tap_span();
        let mut times = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            self.take_gpu_time();
            self.forward(anchor, true)?;
            times.push(self.take_gpu_time());
            if !self.restore_prompt_checkpoint(&checkpoint)? {
                return Err(crate::Error::Generation(
                    "the cost table's checkpoint was purged".into(),
                ));
            }
        }
        costs[0] = median(times);
        for rows in 2..=MAX_TREE_ROWS {
            let tree = DraftTree::from_parents(&vec![anchor; rows], &COST_PARENTS[..rows])?;
            let mut times = Vec::with_capacity(repetitions);
            for _ in 0..repetitions {
                self.take_gpu_time();
                self.tree_verify(&tree)?;
                times.push(self.take_gpu_time());
            }
            costs[rows - 1] = median(times);
        }
        if let Some(state) = self.tree.as_mut() {
            state.pending = None;
        }
        // The restores forgot the taps, which still hold every committed
        // position but the one the decode steps overwrote.
        self.trim_taps(span);
        self.take_gpu_time();
        debug_assert_eq!(self.position, start);
        Ok(costs)
    }

    /// The tree pass: [`Self::encode_block`]'s verify block with the tree's
    /// rotary positions, attention and recurrence.
    #[expect(clippy::too_many_lines)]
    #[expect(unsafe_code)]
    fn encode_tree(&mut self, state: &mut TreeVerifier, tree: &DraftTree) -> crate::Result<()> {
        let rows = tree.len();
        let count = rows as u32;
        if rows > self.block_rows
            || self
                .position
                .checked_add(rows)
                .is_none_or(|end| end > self.info.context)
        {
            return Err(crate::Error::ContextOverflow(
                "invalid Bonsai tree or position".into(),
            ));
        }
        self.selection = Selection::None;
        self.distributions.described = Described::None;
        self.candidates.described = Described::None;
        self.reserve_kv(self.position + rows, None)?;
        let partial_bytes =
            self.ops
                .tree_partial_bytes(self.kv_layout, self.position as u32, TREE_ROWS);
        if state.partials.length() < partial_bytes {
            state.partials = MetalBuffer::empty(self.context.device(), partial_bytes)?;
        }
        decode_embeddings(
            &self.package,
            &tree.tokens,
            // SAFETY: every block waits for its GPU work before returning, and
            // this pass is encoded only after the embeddings are written.
            unsafe { self.scratch.embedding.as_mut_slice::<f32>() },
            None,
        )?;
        // Each row rotates by its own position, the anchor's plus its depth.
        let rope = tree
            .depths
            .iter()
            .map(|&depth| {
                let position = self.position + depth as usize;
                self.images.as_deref().map_or_else(
                    || [position as u32, position as u32, position as u32, 0],
                    |images| images.rope(position),
                )
            })
            .collect::<Vec<_>>();
        self.scratch
            .rope
            .copy_from_bytes(bytemuck::cast_slice(&rope), 0);
        let mut words = [0i32; TREE_DESCRIPTOR_WORDS];
        words[..rows].copy_from_slice(&tree.parents);
        for (word, mask) in words[TREE_MASKS..].iter_mut().zip(tree.masks()) {
            *word = mask.cast_signed();
        }
        state
            .descriptor
            .copy_from_bytes(bytemuck::cast_slice(&words), 0);

        let mut batch = CommandBatch::new_concurrent(&self.context)?;
        let scratch = &self.scratch;
        self.kernels.transform(
            &mut batch,
            &self.input_rotation,
            &scratch.embedding,
            &scratch.hidden,
            count,
            HadamardDirection::Inverse,
        )?;
        let mut recurrent_index = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            if SUBMIT_BEFORE_LAYERS.contains(&index) {
                batch.submit_and_renew(&self.context)?;
            }
            if index == 0 {
                self.normalize_input(&mut batch, &scratch.hidden, &layer.attention_norm, count)?;
            } else {
                self.residual_normalize(
                    &mut batch,
                    &scratch.hidden_alt,
                    &scratch.hidden,
                    &layer.attention_norm,
                    count,
                )?;
            }
            self.encode_taps(&mut batch, index, rows, true)?;
            match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => {
                    self.recurrent_tree(
                        &mut batch,
                        recurrent,
                        recurrent_index,
                        count,
                        &state.descriptor,
                    )?;
                    recurrent_index += 1;
                }
                AttentionLayer::Full(full) => {
                    self.full_attention_tree(&mut batch, full, count, state)?;
                }
            }
            self.residual_normalize(
                &mut batch,
                &scratch.hidden,
                &scratch.hidden_alt,
                &layer.post_attention_norm,
                count,
            )?;
            self.feed_forward_branch(&mut batch, layer, count)?;
        }
        // Every recurrent layer folded the tape's pending rows in.
        self.tape.clear();
        self.residual_normalize(
            &mut batch,
            &scratch.hidden_alt,
            &scratch.hidden,
            &self.output_norm,
            count,
        )?;
        let logits = &state.verifier.verify_logits;
        self.project(
            &mut batch,
            &self.output,
            &scratch.rotated_hidden,
            logits,
            count,
        )?;
        self.greedy.encode(&mut batch, logits, rows)?;
        if let Some(k) = self.top_k {
            self.candidates.reserve(&self.context, rows)?;
            self.candidates.encode(&mut batch, logits, rows, k)?;
            self.candidates.described = Described::Rows(rows);
            self.candidates.k = k;
        }
        if let Some(top) = self.logprobs {
            self.distributions.reserve(&self.context, rows)?;
            self.distributions.encode(&mut batch, logits, rows, top)?;
            self.distributions.described = Described::Rows(rows);
            self.distributions.top = top;
        }
        self.finish(batch)?;
        if self
            .greedy
            .results(rows)
            .iter()
            .any(|result| result.nonfinite != 0)
        {
            return Err(crate::Error::Generation("non-finite Bonsai logits".into()));
        }
        if self.device_greedy {
            self.selection = Selection::Rows(rows);
        }
        Ok(())
    }

    /// [`Self::recurrent`] for a tree: the raw QKV rows, decay and beta go
    /// to the tape for the commit, the convolution and recurrence follow
    /// each row's path, and the layer's state and history are only read,
    /// once the tape's pending rows are folded in. `index` is the layer's
    /// place among the recurrent layers.
    fn recurrent_tree(
        &self,
        batch: &mut CommandBatch,
        layer: &RecurrentLayer,
        index: usize,
        tokens: u32,
        tree: &MetalBuffer,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        let rollback = self.tape.step(index);
        let fold =
            (self.tape.pending() > 0).then(|| self.tape.fold(index, (&self.tape_scratch, 0)));
        batch.independent(|batch| {
            self.project(
                batch,
                &layer.qkv,
                &scratch.rotated_hidden,
                &rollback.raw,
                tokens,
            )?;
            self.project(
                batch,
                &layer.gate,
                &scratch.rotated_hidden,
                &scratch.gate,
                tokens,
            )?;
            self.ops.bf16_matmul_pair(
                batch,
                [
                    (decay_projection(&layer.alpha), &scratch.alpha),
                    (decay_projection(&layer.beta), &scratch.raw_beta),
                ],
                &scratch.normalized,
                tokens,
            )?;
            crate::Result::Ok(())
        })?;
        let decay_inputs = [&layer.decay, &scratch.alpha, &layer.dt, &scratch.raw_beta];
        if let Some(fold) = fold {
            self.ops.conv_l2_decay_tree_fold(
                batch,
                &rollback.raw,
                &layer.convolution,
                &layer.history,
                &scratch.convolved,
                self.epsilon,
                decay_inputs,
                [&rollback.decay, &rollback.beta],
                tree,
                fold,
                tokens,
            )?;
            self.ops.gdn_tree_fold(
                batch,
                self.state_format,
                &scratch.convolved,
                &rollback.decay,
                &rollback.beta,
                &layer.state,
                &scratch.recurrent_output,
                tree,
                fold,
                tokens,
            )?;
        } else {
            self.ops.conv_l2_decay_tree(
                batch,
                &rollback.raw,
                &layer.convolution,
                &layer.history,
                &scratch.convolved,
                self.epsilon,
                decay_inputs,
                &rollback.decay,
                &rollback.beta,
                tree,
                tokens,
            )?;
            self.ops.gdn_tree(
                batch,
                self.state_format,
                &scratch.convolved,
                &rollback.decay,
                &rollback.beta,
                &layer.state,
                &scratch.recurrent_output,
                tree,
                tokens,
            )?;
        }
        self.ops.gdn_postprocess_transform(
            batch,
            &scratch.recurrent_output,
            &scratch.gate,
            &layer.norm,
            &self.attention_rotation,
            &scratch.rotated_attention,
            self.epsilon,
            tokens,
        )?;
        self.project(
            batch,
            &layer.output,
            &scratch.rotated_attention,
            &scratch.branch,
            tokens,
        )
    }

    /// [`Self::full_attention`] for a tree: K/V rows at cache rows
    /// `position + row` with the tree's rotary rows, then tree attention.
    fn full_attention_tree(
        &self,
        batch: &mut CommandBatch,
        layer: &FullAttentionLayer,
        tokens: u32,
        state: &TreeVerifier,
    ) -> crate::Result<()> {
        let scratch = &self.scratch;
        batch.independent(|batch| {
            for (weights, output) in [
                (&layer.query_gate, &scratch.query_gate),
                (&layer.key, &scratch.key),
                (&layer.value, &scratch.value),
            ] {
                self.project(batch, weights, &scratch.rotated_hidden, output, tokens)?;
            }
            crate::Result::Ok(())
        })?;
        self.ops.prepare_attention_rows_kv(
            self.kv_layout,
            batch,
            &scratch.query_gate,
            &scratch.key,
            &scratch.value,
            &layer.query_norm,
            &layer.key_norm,
            &scratch.query,
            &scratch.gate,
            &layer.key_cache,
            &layer.value_cache,
            self.position as u32,
            self.kv_allocated as u32,
            self.epsilon,
            self.rope_base,
            tokens,
            Rope::Rows(&scratch.rope, 0),
        )?;
        self.ops.attention_tree_kv(
            self.kv_layout,
            batch,
            &scratch.query,
            &layer.key_cache,
            &layer.value_cache,
            Some(&scratch.gate),
            &scratch.attention_output,
            self.position as u32,
            tokens,
            &state.descriptor,
            &state.partials,
            &self.scratch.attention,
        )?;
        self.project_attention(batch, &layer.output, tokens)
    }

    /// Drop a pending tree verify (a new request, or a sequence swap).
    pub(super) fn drop_tree_verify(&mut self) {
        if let Some(state) = self.tree.as_mut() {
            state.pending = None;
        }
    }
}

#[cfg(test)]
mod tests;
