//! Reading and verifying a disk snapshot off the engine thread.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crate::bonsai_native::PromptSnapshot;

/// A disk snapshot being read and SHA-256-checked on a thread of its own.
///
/// Both scale with the prompt (about 38 KiB a token), so on the engine thread
/// they held up every decoding request for the whole read. The engine starts
/// one of these instead, keeps the request out of prefill until
/// [`Self::is_finished`], and then only copies the snapshot into the model's
/// buffers. Dropping a load still running stops its read within one piece.
pub struct Load {
    path: PathBuf,
    abort: Arc<AtomicBool>,
    thread: Option<JoinHandle<crate::Result<PromptSnapshot>>>,
}

impl Load {
    /// Start reading the snapshot at `path`, written for `model_key`.
    pub fn spawn(path: PathBuf, model_key: String) -> std::io::Result<Self> {
        let abort = Arc::new(AtomicBool::new(false));
        let thread = std::thread::Builder::new()
            .name("prompt-cache-load".into())
            .spawn({
                let path = path.clone();
                let abort = Arc::clone(&abort);
                move || super::load_abortable(&path, &model_key, &abort)
            })?;
        Ok(Self {
            path,
            abort,
            thread: Some(thread),
        })
    }

    /// The snapshot file being read.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the read has ended, so [`Self::wait`] returns at once.
    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// The verified snapshot, waiting for the rest of the read if needed.
    pub fn wait(mut self) -> crate::Result<PromptSnapshot> {
        let thread = self.thread.take().ok_or_else(|| {
            crate::Error::Io(std::io::Error::other("prompt cache load already taken"))
        })?;
        thread.join().unwrap_or_else(|_| {
            Err(crate::Error::Io(std::io::Error::other(
                "prompt cache load thread panicked",
            )))
        })
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.abort.store(true, Ordering::Release);
        }
    }
}
