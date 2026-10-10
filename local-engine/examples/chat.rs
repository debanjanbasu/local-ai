use std::io::Write as _;
use std::ops::ControlFlow;

use local_engine::{ChatMessage, ChatRequest, Engine, Event, Sampling};

fn main() -> local_engine::Result<()> {
    let mut engine = match Engine::open() {
        Ok(engine) => engine,
        Err(local_engine::Error::InvalidArgument(message))
            if message.contains("model not found") =>
        {
            eprintln!("model is not installed; skipping example");
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let request = ChatRequest {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Say hello in one sentence.".into(),
            ..ChatMessage::default()
        }],
        max_tokens: 128,
        sampling: Sampling::default(),
        thinking: true,
        session: None,
        tools: Vec::new(),
        response_format: local_engine::ResponseFormat::Text,
    };
    engine.chat_with(&request, |event| {
        if let Event::Content(text) = event {
            print!("{text}");
            let _ = std::io::stdout().flush();
        }
        ControlFlow::Continue(())
    })?;
    println!();
    Ok(())
}
