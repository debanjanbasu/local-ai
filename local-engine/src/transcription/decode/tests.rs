#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation
)]

use std::collections::BTreeMap;

use super::*;
use crate::transcription::tokens::{LANGUAGES, SpecialTokens, WhisperTokenizer};

const EOT: u32 = 50_257;
const SOT: u32 = 50_258;
const TS: u32 = 50_365;
const VOCAB: usize = 51_866;

fn special() -> SpecialTokens {
    let languages: BTreeMap<String, u32> = LANGUAGES
        .iter()
        .enumerate()
        .map(|(i, (code, _))| ((*code).to_owned(), SOT + 1 + i as u32))
        .collect();
    SpecialTokens {
        sot: SOT,
        eot: EOT,
        sot_prev: 50_362,
        no_timestamps: 50_364,
        transcribe: 50_360,
        translate: 50_359,
        sot_lm: 50_361,
        no_speech: 50_363,
        timestamp_begin: TS,
        languages,
        suppress_tokens: vec![1, 2],
        begin_suppress_tokens: vec![220, EOT],
        alignment_heads: vec![],
        max_initial_timestamp_index: 50,
    }
}

fn tokenizer() -> WhisperTokenizer {
    let bytes = (0..EOT).map(|id| format!("w{id} ").into_bytes()).collect();
    WhisperTokenizer::from_bytes(bytes, special())
}

fn ts(seconds: f64) -> u32 {
    TS + (seconds / 0.02).round() as u32
}

/// Emits scripted tokens: one script per decode attempt (prefill call).
struct Script {
    attempts: Vec<Vec<u32>>,
    attempt: usize,
    position: usize,
    no_speech_logit: f32,
    flat: bool,
    encoded: usize,
}

impl Script {
    fn new(attempts: Vec<Vec<u32>>) -> Self {
        Self {
            attempts,
            attempt: 0,
            position: 0,
            no_speech_logit: 0.0,
            flat: false,
            encoded: 0,
        }
    }

    fn logits(&mut self) -> Vec<f32> {
        let mut logits = vec![0.0_f32; VOCAB];
        if !self.flat {
            let script = &self.attempts[self.attempt - 1];
            let token = script.get(self.position).copied().unwrap_or(EOT);
            logits[token as usize] = 50.0;
        }
        self.position += 1;
        logits
    }
}

impl Forward for Script {
    fn encode(&mut self, window: &[f32], _: &mut Pace<'_>) -> crate::Result<()> {
        assert_eq!(window.len(), WINDOW_FRAMES * 2);
        self.encoded += 1;
        Ok(())
    }

    fn prefill(
        &mut self,
        _tokens: &[u32],
        rows: &[usize],
        _: &mut Pace<'_>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        assert_eq!(rows.len(), 2);
        self.attempt += 1;
        self.position = 0;
        let mut at_sot = vec![0.0_f32; VOCAB];
        at_sot[50_363] = self.no_speech_logit;
        Ok(vec![at_sot, self.logits()])
    }

    fn step(&mut self, _token: u32, _: &mut Pace<'_>) -> crate::Result<Vec<f32>> {
        Ok(self.logits())
    }
}

fn mel(content_frames: usize) -> LogMel {
    let frames = content_frames + WINDOW_FRAMES;
    LogMel {
        n_mels: 2,
        frames,
        data: vec![0.0; 2 * frames],
    }
}

fn run(
    script: &mut Script,
    content_frames: usize,
    temperature: Temperature,
) -> (Vec<Segment>, Vec<u32>, Work) {
    let tokenizer = tokenizer();
    let options = WindowOptions {
        task: Task::Transcribe,
        language: SOT + 1,
        temperature,
        condition_on_previous_text: true,
    };
    let mut work = Work::default();
    let busy = || false;
    let mut pause = || Ok(());
    let mut pace = Pace {
        busy: &busy,
        pause: &mut pause,
    };
    let mut seen = 0;
    let (segments, text) = transcribe(
        script,
        &tokenizer,
        &mel(content_frames),
        false,
        &options,
        &[],
        &mut Rng::new(7),
        &mut pace,
        &mut work,
        &mut |segments| {
            seen += segments.len();
            Ok(())
        },
    )
    .expect("transcribe");
    assert_eq!(seen, segments.len());
    (segments, text, work)
}

#[test]
fn consecutive_timestamps_split_segments_and_a_single_ending_consumes_the_window() {
    let mut script = Script::new(vec![vec![
        ts(0.0),
        10,
        11,
        ts(2.0),
        ts(2.0),
        12,
        13,
        ts(5.0),
        EOT,
    ]]);
    let (segments, text, work) = run(&mut script, 1000, Temperature::Fixed(0.0));
    assert_eq!(work.windows, 1);
    assert_eq!(segments.len(), 2);
    assert_eq!((segments[0].start, segments[0].end), (0.0, 2.0));
    assert_eq!(segments[0].text, "w10 w11 ");
    assert_eq!(segments[0].tokens, vec![ts(0.0), 10, 11, ts(2.0)]);
    assert_eq!((segments[1].start, segments[1].end), (2.0, 5.0));
    assert_eq!(segments[1].id, 1);
    assert_eq!(
        text,
        vec![ts(0.0), 10, 11, ts(2.0), ts(2.0), 12, 13, ts(5.0)]
    );
}

#[test]
fn an_unfinished_segment_is_dropped_and_the_next_window_starts_at_the_last_timestamp() {
    let mut script = Script::new(vec![
        vec![ts(0.0), 10, ts(1.0), ts(1.0), 11, ts(3.0), ts(3.0), 12, EOT],
        vec![ts(0.0), 12, ts(1.0), EOT],
    ]);
    let (segments, _, work) = run(&mut script, 1000, Temperature::Fixed(0.0));
    assert_eq!(work.windows, 2);
    assert_eq!(segments.len(), 3);
    assert_eq!((segments[1].start, segments[1].end), (1.0, 3.0));
    // The second window starts at frame 300 (3 s) and ends with a single
    // timestamp, so it consumes the rest of the audio.
    assert_eq!(segments[2].seek, 300);
    assert_eq!((segments[2].start, segments[2].end), (3.0, 4.0));
    assert_eq!(segments[2].text, "w12 ");
}

#[test]
fn text_without_timestamp_pairs_is_one_segment_up_to_its_last_timestamp() {
    let mut script = Script::new(vec![vec![ts(0.0), 10, 11, EOT]]);
    let (segments, _, _) = run(&mut script, 800, Temperature::Fixed(0.0));
    assert_eq!(segments.len(), 1);
    // The only timestamp is <|0.00|>, so the segment spans the window content.
    assert_eq!((segments[0].start, segments[0].end), (0.0, 8.0));
}

#[test]
fn repetitive_output_falls_back_to_a_higher_temperature() {
    let mut repetitive = vec![ts(0.0)];
    repetitive.extend(std::iter::repeat_n(10, 120));
    repetitive.push(ts(9.0));
    let mut script = Script::new(vec![repetitive, vec![ts(0.0), 10, 11, ts(2.0), EOT]]);
    let (segments, _, work) = run(&mut script, 1000, Temperature::Fallback);
    assert_eq!(script.attempt, 2);
    assert!(work.sampled_tokens > 120);
    assert_eq!(segments[0].temperature, 0.2);
    assert_eq!(segments[0].text, "w10 w11 ");
}

#[test]
fn likely_silence_with_unlikely_text_is_skipped() {
    let mut script = Script::new(vec![vec![]]);
    script.flat = true;
    script.no_speech_logit = 30.0;
    let (segments, text, work) = run(&mut script, 1000, Temperature::Fallback);
    // No fallback (silence), then the window is skipped.
    assert_eq!(script.attempt, 1);
    assert_eq!(work.windows, 1);
    assert_eq!(segments, Vec::<Segment>::new());
    assert_eq!(text, Vec::<u32>::new());
}

#[test]
fn timestamp_rules_follow_the_reference() {
    let tokenizer = tokenizer();
    let suppress = tokenizer.special.suppressed();
    let begin = 4;
    let tokens = vec![50_362, SOT, SOT + 1, 50_360];
    // At the start only timestamps up to 1.00 s are allowed.
    let mut logits = vec![0.0_f32; VOCAB];
    apply_filters(&mut logits, &tokens, begin, &tokenizer, &suppress);
    assert!(logits[..TS as usize].iter().all(|v| v.is_infinite()));
    assert!(logits[ts(1.0) as usize].is_finite());
    assert!(logits[ts(1.02) as usize].is_infinite());
    // After text then a timestamp: no text, only timestamps >= it, or EOT.
    let mut sampled = tokens;
    sampled.extend([ts(0.0), 10, ts(1.5)]);
    let mut logits = vec![0.0_f32; VOCAB];
    logits[EOT as usize] = 20.0; // dominant, so the timestamp mass does not force a timestamp
    apply_filters(&mut logits, &sampled, begin, &tokenizer, &suppress);
    assert!(logits[10].is_infinite());
    assert!(logits[EOT as usize].is_finite());
    assert!(logits[ts(1.48) as usize].is_infinite());
    assert!(logits[ts(1.5) as usize].is_finite());
    // After a timestamp pair: text only, and later timestamps must increase.
    sampled.push(ts(1.5));
    let mut logits = vec![0.0_f32; VOCAB];
    logits[10] = 5.0;
    apply_filters(&mut logits, &sampled, begin, &tokenizer, &suppress);
    assert!(logits[TS as usize..].iter().all(|v| v.is_infinite()));
    assert!(logits[10].is_finite());
    assert!(logits[1].is_infinite() && logits[50_359].is_infinite());
}

#[test]
fn compression_ratio_flags_repetition() {
    assert!(compression_ratio(&"the same words again ".repeat(30)) > 2.4);
    assert!(compression_ratio("A short, varied sentence.") < 1.5);
    assert_eq!(compression_ratio(""), 0.0);
}
