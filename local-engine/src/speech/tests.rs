use super::*;

fn token(text: &str, phonemes: &str, whitespace: bool) -> Token {
    Token {
        text: text.to_owned(),
        phonemes: Some(phonemes.to_owned()),
        whitespace,
    }
}

#[test]
fn chunks_cut_at_the_last_sentence_end_that_fits() {
    let tokens = vec![
        token("aaa", "aaa", false),
        token(".", ".", true),
        token("bbb", "bbb", false),
        token(",", ",", true),
        token("ccc", "ccc", false),
        token(".", ".", false),
    ];
    let chunks = chunk_tokens(&tokens, 10);
    let phonemes: Vec<&str> = chunks.iter().map(|c| c.phonemes.as_str()).collect();
    assert_eq!(phonemes, ["aaa.", "bbb, ccc."]);
    assert!(chunks.iter().all(|c| c.phonemes.chars().count() <= 10));
}

#[test]
fn short_text_is_one_chunk() {
    let tokens = vec![token("hi", "hˈI", false), token("!", "!", false)];
    let chunks = chunk_tokens(&tokens, MAX_CHUNK_PHONEMES);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].phonemes, "hˈI!");
    assert_eq!(chunks[0].text, "hi!");
}

#[test]
fn voice_aliases_are_distinct() {
    let mut ids: Vec<&str> = VOICE_ALIASES.iter().map(|(_, id)| *id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), VOICE_ALIASES.len());
}

#[test]
fn split_half_cuts_at_a_space_near_the_middle() {
    assert_eq!(
        split_half("ab cd ef gh"),
        ("ab cd".to_owned(), "ef gh".to_owned())
    );
}

#[derive(serde::Deserialize)]
struct Reference {
    synthesis: Vec<SynthesisRow>,
    g2p: Vec<G2pRow>,
}

#[derive(serde::Deserialize)]
struct SynthesisRow {
    text: String,
    voice: String,
    speed: f32,
    phonemes: String,
    input_ids: Vec<u32>,
    pred_dur: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct G2pRow {
    lang: String,
    text: String,
    phonemes: String,
}

fn reference() -> Reference {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/kokoro-reference.json");
    serde_json::from_slice(&std::fs::read(path).expect("fixture")).expect("fixture json")
}

fn installed() -> Option<PathBuf> {
    installed_model(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../models"))
}

/// Needs `models/kokoro-82m` (CPU only): the native G2P reproduces `misaki`
/// 0.9.4 on every fixture text in both accents.
#[test]
#[ignore = "needs the installed Kokoro-82M artifact"]
fn native_g2p_matches_misaki_on_the_fixture_texts() {
    let dir = installed().expect("models/kokoro-82m is not installed");
    let (american, british) = open_g2p(&dir).expect("g2p");
    let fixture = reference();
    for row in &fixture.g2p {
        let g2p = if row.lang == "b" { &british } else { &american };
        assert_eq!(
            g2p.phonemize(&row.text, false).trim_end(),
            row.phonemes,
            "{}",
            row.text
        );
    }
}

/// Needs `models/kokoro-82m` and a Metal device: ids and durations of the
/// reference phonemes match the reference.
#[test]
#[ignore = "needs the installed Kokoro-82M artifact and a Metal device"]
fn native_durations_match_the_reference_fixture() {
    let dir = installed().expect("models/kokoro-82m is not installed");
    let model = SpeechModel::open(&dir).expect("model");
    let fixture = reference();
    let mut same = 0;
    for row in &fixture.synthesis {
        assert_eq!(model.ids(&row.phonemes), row.input_ids, "{}", row.text);
        let out = model
            .synthesize_phonemes(
                &row.phonemes,
                &row.voice,
                row.speed,
                Noise::Zero,
                &mut || Ok(()),
                None,
            )
            .expect("synthesis");
        assert_eq!(
            out.samples.len(),
            600 * out.durations.iter().sum::<u32>() as usize
        );
        same += usize::from(out.durations == row.pred_dur);
    }
    assert!(
        same * 10 >= fixture.synthesis.len() * 9,
        "{same} of {} identical",
        fixture.synthesis.len()
    );
}
