use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLFunction, MTLLibrary};

use crate::Error;

/// Pre-compiled Metal shader library loaded from the embedded metallib.
pub struct ShaderLibrary {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl ShaderLibrary {
    /// Load the embedded shader library onto the given device.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LibraryCreation`] if the metallib data cannot be loaded.
    #[allow(unsafe_code)]
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<Self> {
        let metallib_bytes: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
        let data = dispatch2::DispatchData::from_bytes(metallib_bytes);
        let library = device
            .newLibraryWithData_error(&data)
            .map_err(|e| Error::LibraryCreation(e.to_string()))?;
        Ok(Self { library })
    }

    /// Look up a kernel function by name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShaderNotFound`] if the function does not exist.
    pub fn get_function(
        &self,
        name: &str,
    ) -> crate::Result<Retained<ProtocolObject<dyn MTLFunction>>> {
        let ns_name = NSString::from_str(name);
        self.library
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| Error::ShaderNotFound(name.to_owned()))
    }
}
