use std::path::Path;

impl crate::pipeline::Pipeline {
    pub fn new(model_dir: &Path, max_context_length: usize) -> crate::Result<Self> {
        Self::new_qat(model_dir, max_context_length)
    }
}
