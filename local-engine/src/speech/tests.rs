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
    assert_eq!(split_half("ab cd ef gh"), ("ab cd".to_owned(), "ef gh".to_owned()));
}
