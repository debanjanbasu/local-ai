//! Pipelined greedy plain decode: the pass of token N+1 is committed before
//! the host waits for token N, its input read on the GPU from token N's
//! argmax, so the GPU never idles between tokens.

use local_metal::batch::PendingCommandBuffer;
use local_metal::draft::{DraftKernels, GreedyRows};

use super::layers::RecurrentTarget;
use super::{
    AttentionLayer, BonsaiMetalTensor, BonsaiModel, CONV_STATE_BYTES, CommandBatch, Described,
    HadamardDirection, MetalBuffer, Rope, Sampler, SamplingResult, Selection, ShaderLibrary, VOCAB,
    decode_embeddings,
};

/// Submit before these layers, as [`super::block`] does.
const SUBMIT_BEFORE_LAYERS: [usize; 3] = [0, 1, 4];

/// What a pipelined pass needs besides the model's scratch.
pub(super) struct Pipeline {
    kernels: DraftKernels,
    embeddings: BonsaiMetalTensor,
    /// Two argmax selectors, used in turn: the pass in flight writes one
    /// while the host reads the other.
    selectors: [GreedyRows; 2],
    next_selector: usize,
    /// A second recurrent state and history per GDN layer. A pass reads the
    /// layer's own and writes these, and the two then swap, so the state
    /// before the pass in flight survives until the next pass is committed:
    /// a pass that decoded an end of sequence rolls back by swapping again.
    spare: Vec<(MetalBuffer, MetalBuffer)>,
    in_flight: Option<InFlight>,
}

struct InFlight {
    pending: PendingCommandBuffer,
    selector: usize,
    /// The token the pass decodes and whether it ends the sequence, once the
    /// host has read it from the previous pass: the pass counts only when the
    /// next round decodes exactly that token.
    input: Option<(u32, bool)>,
}

#[derive(Clone, Copy)]
enum PassInput {
    Host(u32),
    Device(usize),
}

impl BonsaiModel {
    /// Whether [`Self::decode_pipelined`] may serve this model's plain
    /// decode: no MTP head to feed and no prompt images to position.
    pub(crate) const fn pipelines_decode(&self) -> bool {
        self.speculation.is_none() && self.images.is_none()
    }

    /// Decode `seed` and select the next token by argmax, leaving the pass
    /// that decodes that token in flight unless it is the `last`. The
    /// sampler must select the argmax with no grammar, log-probabilities or
    /// device top k. To the caller the model is as after [`Self::decode`]:
    /// the pass in flight counts only once the next call decodes the token
    /// it guessed.
    pub(crate) fn decode_pipelined(
        &mut self,
        seed: u32,
        sampler: &mut Sampler,
        last: bool,
    ) -> crate::Result<SamplingResult> {
        let mut pipeline = match self.pipeline.take() {
            Some(pipeline) => pipeline,
            None => self.new_pipeline()?,
        };
        let result = self.pipelined_round(&mut pipeline, seed, sampler, last);
        self.pipeline = Some(pipeline);
        result
    }

    /// Wait for the pass in flight, if any, and roll it back. Every other
    /// use of the model's state must come after this.
    pub(crate) fn settle_decode(&mut self) -> crate::Result<()> {
        let Some(mut pipeline) = self.pipeline.take() else {
            return Ok(());
        };
        let result = pipeline
            .in_flight
            .take()
            .map_or(Ok(()), |flight| self.roll_back(&mut pipeline, flight));
        self.pipeline = Some(pipeline);
        result
    }

    fn new_pipeline(&self) -> crate::Result<Pipeline> {
        let shaders = ShaderLibrary::new(self.context.device())?;
        let device = self.context.device();
        let spare = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.attention, AttentionLayer::Recurrent(_)))
            .map(|_| {
                Ok((
                    MetalBuffer::empty(device, self.state_format.state_bytes())?,
                    MetalBuffer::empty(device, CONV_STATE_BYTES)?,
                ))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Pipeline {
            kernels: DraftKernels::new(&self.context, &shaders, VOCAB)?,
            embeddings: self.package.metal_tensor(device, "token_embd.weight")?,
            selectors: [
                GreedyRows::new(&self.context, &shaders, VOCAB, 1)?,
                GreedyRows::new(&self.context, &shaders, VOCAB, 1)?,
            ],
            next_selector: 0,
            spare,
            in_flight: None,
        })
    }

    fn pipelined_round(
        &mut self,
        pipeline: &mut Pipeline,
        seed: u32,
        sampler: &mut Sampler,
        last: bool,
    ) -> crate::Result<SamplingResult> {
        let current = match pipeline.in_flight.take() {
            Some(flight) if flight.input == Some((seed, false)) => flight,
            Some(flight) => {
                self.roll_back(pipeline, flight)?;
                self.commit_pass(pipeline, PassInput::Host(seed))?
            }
            None => self.commit_pass(pipeline, PassInput::Host(seed))?,
        };
        // Growing the caches copies rows a pass in flight may still write.
        let next = (!last && self.position < self.kv_allocated)
            .then(|| self.commit_pass(pipeline, PassInput::Device(current.selector)))
            .transpose();
        let waited = current.pending.wait_with_gpu_time();
        let next = match (next, waited) {
            (Ok(next), Ok(gpu)) => {
                self.gpu_time.set(self.gpu_time.get().saturating_add(gpu));
                next
            }
            (Err(error), _) => return Err(error),
            (Ok(next), Err(error)) => {
                if let Some(next) = next {
                    let _ = next.pending.wait();
                }
                return Err(error.into());
            }
        };
        let selected = pipeline.selectors[current.selector].results(1)[0];
        if selected.nonfinite != 0 {
            if let Some(next) = next {
                let _ = next.pending.wait();
            }
            return Err(crate::Error::Generation("non-finite Bonsai logits".into()));
        }
        let sample = sampler.select_row(
            &mut self.scratch.logits,
            0,
            &self.context,
            &self.sampling,
            Some(selected.best_id),
            None,
            None,
        )?;
        if let Some(mut next) = next {
            next.input = Some((sample.token_id, sample.is_eos));
            pipeline.in_flight = Some(next);
        }
        Ok(sample)
    }

    /// Wait for `flight` and undo it: the state it read is still in the
    /// spare buffers, since no pass was committed after it.
    fn roll_back(&mut self, pipeline: &mut Pipeline, flight: InFlight) -> crate::Result<()> {
        let gpu = flight.pending.wait_with_gpu_time()?;
        self.gpu_time.set(self.gpu_time.get().saturating_add(gpu));
        self.swap_spare(pipeline);
        self.position -= 1;
        Ok(())
    }

    fn swap_spare(&mut self, pipeline: &mut Pipeline) {
        let recurrent = self
            .layers
            .iter_mut()
            .filter_map(|layer| match &mut layer.attention {
                AttentionLayer::Recurrent(recurrent) => Some(recurrent),
                AttentionLayer::Full(_) => None,
            });
        for (layer, (state, history)) in recurrent.zip(&mut pipeline.spare) {
            std::mem::swap(&mut layer.state, state);
            std::mem::swap(&mut layer.history, history);
        }
    }

    /// Encode and commit one decode pass at the current position, writing
    /// the recurrent state into the spare buffers, and swap them in.
    #[allow(unsafe_code, clippy::too_many_lines)]
    fn commit_pass(
        &mut self,
        pipeline: &mut Pipeline,
        input: PassInput,
    ) -> crate::Result<InFlight> {
        if self.position >= self.info.context {
            return Err(crate::Error::ContextOverflow(
                "invalid Bonsai token block or position".into(),
            ));
        }
        self.reserve_kv(self.position + 1, None)?;
        self.selection = Selection::None;
        self.distributions.described = Described::None;
        self.candidates.described = Described::None;
        let selector = pipeline.next_selector;
        pipeline.next_selector ^= 1;
        if let PassInput::Host(token) = input {
            crate::bonsai_vision::check_tokens(&[token], self.position, None)?;
            decode_embeddings(
                &self.package,
                &[token],
                // SAFETY: host input is only encoded with no pass in flight,
                // so nothing on the GPU reads the embedding.
                unsafe { self.scratch.embedding.as_mut_slice::<f32>() },
                None,
            )?;
        }
        let mut batch = CommandBatch::new(&self.context)?;
        let scratch = &self.scratch;
        match input {
            PassInput::Host(_) => {
                self.kernels.transform(
                    &mut batch,
                    &self.input_rotation,
                    &scratch.embedding,
                    &scratch.hidden,
                    1,
                    HadamardDirection::Inverse,
                )?;
            }
            PassInput::Device(source) => pipeline.kernels.embed_inverse(
                &mut batch,
                pipeline.embeddings.ptq1_matrix()?,
                &self.input_rotation,
                pipeline.selectors[source].results_buffer(),
                0,
                &scratch.hidden,
            )?,
        }
        let mut recurrent_index = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            if SUBMIT_BEFORE_LAYERS.contains(&index) {
                batch.submit_and_renew(&self.context)?;
            }
            if index == 0 {
                self.normalize_input(&mut batch, &scratch.hidden, &layer.attention_norm, 1)?;
            } else {
                self.residual_normalize(
                    &mut batch,
                    &scratch.hidden_alt,
                    &scratch.hidden,
                    &layer.attention_norm,
                    1,
                )?;
            }
            match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => {
                    let (state, history) = &pipeline.spare[recurrent_index];
                    recurrent_index += 1;
                    self.recurrent(
                        &mut batch,
                        recurrent,
                        1,
                        RecurrentTarget::Into { state, history },
                    )?;
                }
                AttentionLayer::Full(full) => {
                    self.full_attention(&mut batch, full, 1, Rope::Text(0))?;
                }
            }
            self.residual_normalize(
                &mut batch,
                &scratch.hidden,
                &scratch.hidden_alt,
                &layer.post_attention_norm,
                1,
            )?;
            self.feed_forward_branch(&mut batch, layer, 1)?;
        }
        self.residual_normalize(
            &mut batch,
            &scratch.hidden_alt,
            &scratch.hidden,
            &self.output_norm,
            1,
        )?;
        self.project(
            &mut batch,
            &self.output,
            &scratch.rotated_hidden,
            &scratch.logits,
            1,
        )?;
        pipeline.selectors[selector].encode(&mut batch, &scratch.logits, 1)?;
        let pending = batch.commit_async();
        self.swap_spare(pipeline);
        self.position += 1;
        Ok(InFlight {
            pending,
            selector,
            input: None,
        })
    }
}
