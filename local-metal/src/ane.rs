//! Client of Apple's private `AppleNeuralEngine.framework`, the Neural Engine
//! ("ANE"), for the Bonsai prefill FFN (`local-engine`'s `ane_split`).
//!
//! Adapted from Kapitsa's `src/ane/ane.{h,mm}` (nace-ai/kapitsa at `a6a7bca`,
//! Apache-2.0), whose interface declarations and signature table come from
//! Splash (incoai/splash, Apache-2.0); see `THIRD_PARTY_NOTICES.md`.
//!
//! - Every selector is checked against its arm64 type encoding before the
//!   first call, so a changed method is refused, never mis-called.
//! - Programs are MIL text whose weights are IOSurface-backed
//!   `tensor_buffer` inputs, written under
//!   `$(DARWIN_USER_CACHE_DIR)/<tag>/<hash>/`. The ANE daemon keeps
//!   compilations under that hash; one it holds but cannot load is purged
//!   and compiled again.
//! - `IOSurface`s are shared with Metal without copies ([`Surface`]).
//! - Evaluations are ordered with the GPU by two `MTLSharedEvent`s
//!   ([`Handoff`]), with a watchdog that releases the GPU's waits when the
//!   ANE stops answering.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, mpsc};
use std::time::{Duration, Instant};

use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyClass, AnyObject, Bool, ProtocolObject, Sel};
use objc2::{msg_send, sel};
use objc2_core_foundation::{CFDictionary, CFRetained, CFString};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use objc2_io_surface::{
    IOSurfaceRef, kIOSurfaceAllocSize, kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow,
    kIOSurfaceHeight, kIOSurfacePixelFormat, kIOSurfaceWidth,
};
use objc2_metal::{MTLDevice, MTLSharedEvent, MTLSharedEventListener};

use crate::Error;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;

const FRAMEWORK: &CStr =
    c"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
/// `RTLD_NOW` on Darwin.
const RTLD_NOW: i32 = 2;
/// `_CS_DARWIN_USER_CACHE_DIR`.
const CS_DARWIN_USER_CACHE_DIR: i32 = 65_538;
/// `QOS_CLASS_DEFAULT`, Kapitsa's choice for every service call.
const QOS: u32 = 0x15;
/// 'L00h' and 'L008': one-channel fp16 and int8 `IOSurface`s.
const PIXEL_FP16: u32 = 0x4c30_3068;
const PIXEL_INT8: u32 = 0x4c30_3038;
/// `IOSurface`s are sized in whole 16 KiB pages, so Metal can wrap them.
const PAGE: usize = 16_384;

#[expect(unsafe_code)]
unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn confstr(name: i32, buffer: *mut c_char, length: usize) -> usize;
}

fn ane_error(message: impl Into<String>) -> Error {
    Error::NeuralEngine(message.into())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// An Objective-C object shared across threads. The `_ANE*` client objects
/// and Metal shared events are used from the daemon's callback threads by
/// Apple's own clients; Kapitsa uses them the same way.
struct Shared<T: ?Sized>(Retained<T>);

// SAFETY: see the type's documentation; every use is through `&self`
// methods the framework documents (or Kapitsa uses) as thread-safe.
#[expect(unsafe_code)]
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the wrapped objects are thread-safe; see the type's documentation"
)]
unsafe impl<T: ?Sized> Send for Shared<T> {}
// SAFETY: as above.
#[expect(unsafe_code)]
unsafe impl<T: ?Sized> Sync for Shared<T> {}

/// arm64 type encodings with digits stripped: `(class, class method?,
/// selector, encoding)`.
const SIGNATURES: &[(&str, bool, &str, &str)] = &[
    ("_ANEModel", true, "modelAtURL:key:", "@@:@@"),
    ("_ANEModel", false, "modelAttributes", "@@:"),
    ("_ANEClient", true, "sharedConnection", "@@:"),
    (
        "_ANEClient",
        false,
        "compileModel:options:qos:error:",
        "B@:@@I^@",
    ),
    ("_ANEClient", false, "compiledModelExistsFor:", "B@:@"),
    ("_ANEClient", false, "purgeCompiledModel:", "v@:@"),
    (
        "_ANEClient",
        false,
        "loadModel:options:qos:error:",
        "B@:@@I^@",
    ),
    (
        "_ANEClient",
        false,
        "unloadModel:options:qos:error:",
        "B@:@@I^@",
    ),
    (
        "_ANEClient",
        false,
        "evaluateWithModel:options:request:qos:error:",
        "B@:@@@I^@",
    ),
    (
        "_ANEIOSurfaceObject",
        true,
        "objectWithIOSurface:",
        "@@:^{__IOSurface=}",
    ),
    (
        "_ANERequest",
        true,
        "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:",
        "@@:@@@@@@@@@",
    ),
    ("_ANERequest", false, "setCompletionHandler:", "v@:@?"),
    (
        "_ANESharedWaitEvent",
        true,
        "waitEventWithValue:sharedEvent:eventType:",
        "@@:Q@Q",
    ),
    (
        "_ANESharedSignalEvent",
        true,
        "signalEventWithValue:symbolIndex:eventType:sharedEvent:",
        "@@:QIq@",
    ),
    (
        "_ANESharedEvents",
        true,
        "sharedEventsWithSignalEvents:waitEvents:",
        "@@:@@",
    ),
];

struct Api {
    client: Shared<AnyObject>,
    model: &'static AnyClass,
    surface: &'static AnyClass,
    request: &'static AnyClass,
    events: &'static AnyClass,
    signal_event: &'static AnyClass,
    wait_event: &'static AnyClass,
}

fn class(name: &str) -> Result<&'static AnyClass, String> {
    let name = CString::new(name).map_err(|error| error.to_string())?;
    AnyClass::get(&name)
        .ok_or_else(|| format!("AppleNeuralEngine lacks {}", name.to_string_lossy()))
}

#[expect(unsafe_code)]
fn check_signatures() -> Result<(), String> {
    for &(class_name, class_method, selector, expected) in SIGNATURES {
        let class = class(class_name)?;
        let name = CString::new(selector).map_err(|error| error.to_string())?;
        let sel = Sel::register(&name);
        let kind = if class_method { '+' } else { '-' };
        let method = if class_method {
            class.class_method(sel)
        } else {
            class.instance_method(sel)
        }
        .ok_or_else(|| format!("AppleNeuralEngine lacks {kind}[{class_name} {selector}]"))?;
        // SAFETY: `method` is a live runtime method; the encoding it returns is
        // a NUL-terminated string owned by the runtime.
        let encoding = unsafe {
            let raw = objc2::ffi::method_getTypeEncoding(method);
            if raw.is_null() {
                String::new()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        };
        let stripped = encoding
            .chars()
            .filter(|c| !c.is_ascii_digit())
            .collect::<String>();
        if stripped != expected {
            return Err(format!(
                "AppleNeuralEngine changed {kind}[{class_name} {selector}]: {stripped}"
            ));
        }
    }
    Ok(())
}

#[expect(unsafe_code)]
fn resolve() -> Result<Api, String> {
    // SAFETY: dlopen with a constant path; the handle is kept for the process.
    if unsafe { dlopen(FRAMEWORK.as_ptr(), RTLD_NOW) }.is_null() {
        // SAFETY: dlerror returns a NUL-terminated message or null.
        let reason = unsafe {
            let raw = dlerror();
            if raw.is_null() {
                "unknown error".to_owned()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        };
        return Err(format!("AppleNeuralEngine does not load: {reason}"));
    }
    check_signatures()?;
    let client_class = class("_ANEClient")?;
    let client = catch("_ANEClient sharedConnection", || {
        // SAFETY: the selector's encoding was checked above.
        let client: Option<Retained<AnyObject>> =
            unsafe { msg_send![client_class, sharedConnection] };
        client
    })
    .map_err(|error| error.to_string())?
    .ok_or_else(|| "no _ANEClient shared connection".to_owned())?;
    Ok(Api {
        client: Shared(client),
        model: class("_ANEModel")?,
        surface: class("_ANEIOSurfaceObject")?,
        request: class("_ANERequest")?,
        events: class("_ANESharedEvents")?,
        signal_event: class("_ANESharedSignalEvent")?,
        wait_event: class("_ANESharedWaitEvent")?,
    })
}

fn api() -> Result<&'static Api, Error> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| autoreleasepool(|_| resolve()))
        .as_ref()
        .map_err(|reason| ane_error(reason.clone()))
}

/// `None` when the private client resolves with the expected selectors and
/// type encodings (checked once per process); otherwise why it cannot be
/// used.
#[must_use]
pub fn unavailable() -> Option<String> {
    api().err().map(|error| match error {
        Error::NeuralEngine(reason) => reason,
        other => other.to_string(),
    })
}

/// Run `body`, turning an Objective-C exception into an error.
fn catch<R>(what: &str, body: impl FnOnce() -> R) -> crate::Result<R> {
    objc2::exception::catch(AssertUnwindSafe(body)).map_err(|exception| {
        let reason = exception.map_or_else(|| "nil".to_owned(), |e| format!("{e:?}"));
        ane_error(format!("{what} raised {reason}"))
    })
}

fn error_text(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

/// FNV-1a, Kapitsa's program cache key.
#[must_use]
pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    bytes.iter().fold(seed, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
    })
}

/// FNV-1a's offset basis.
pub const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;

/// `$(getconf DARWIN_USER_CACHE_DIR)`.
#[expect(unsafe_code)]
pub fn user_cache_dir() -> crate::Result<PathBuf> {
    let mut buffer = vec![0 as c_char; 1024];
    // SAFETY: the buffer is writable for its length; confstr NUL-terminates.
    let written = unsafe { confstr(CS_DARWIN_USER_CACHE_DIR, buffer.as_mut_ptr(), buffer.len()) };
    if written == 0 || written > buffer.len() {
        return Err(ane_error("no DARWIN_USER_CACHE_DIR"));
    }
    // SAFETY: confstr wrote a NUL-terminated string within the buffer.
    let path = unsafe { CStr::from_ptr(buffer.as_ptr()) };
    Ok(PathBuf::from(path.to_string_lossy().into_owned()))
}

/// Row layout of an `IOSurface` of `rows` x `width` int8 or fp16 elements: rows
/// padded to 64 bytes, size rounded to 16 KiB pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceLayout {
    pub rows: u32,
    pub width: u32,
    pub fp16: bool,
    /// Bytes per row.
    pub stride: usize,
    pub bytes: usize,
}

impl SurfaceLayout {
    #[must_use]
    pub const fn new(rows: u32, width: u32, fp16: bool) -> Self {
        let element = if fp16 { 2 } else { 1 };
        let stride = (width as usize * element).div_ceil(64) * 64;
        let bytes = (stride * rows as usize).div_ceil(PAGE) * PAGE;
        Self {
            rows,
            width,
            fp16,
            stride,
            bytes,
        }
    }

    #[must_use]
    pub const fn element_bytes(self) -> usize {
        if self.fp16 { 2 } else { 1 }
    }

    #[must_use]
    pub const fn mil_type(self) -> &'static str {
        if self.fp16 { "fp16" } else { "int8" }
    }

    /// `[plane, plane, stride, 1]` in elements over all of the surface's rows.
    #[must_use]
    pub fn strides(self) -> String {
        let stride = self.stride / self.element_bytes();
        let plane = stride * self.rows as usize;
        format!("[{plane}, {plane}, {stride}, 1]")
    }

    /// The MIL `tensor_buffer` type of the leading `width` elements of the
    /// first `rows` rows (the surface's strides).
    #[must_use]
    pub fn buffer_type(self, rows: u32, width: u32) -> String {
        format!(
            "tensor_buffer<{}, shape=[1, 1, {rows}, {width}], strides={}, interleave_factors=[1, 1, 1, 1]>",
            self.mil_type(),
            self.strides()
        )
    }
}

/// An owned `IOSurface`.
struct SurfaceRef(CFRetained<IOSurfaceRef>);

// SAFETY: `IOSurface`s are kernel objects meant for sharing across threads
// and processes; this wrapper only reads immutable properties.
#[expect(unsafe_code)]
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "IOSurfaces are thread-safe kernel objects"
)]
unsafe impl Send for SurfaceRef {}
// SAFETY: as above.
#[expect(unsafe_code)]
unsafe impl Sync for SurfaceRef {}

/// An `IOSurface` wrapped zero-copy as a Metal buffer: the GPU writes and reads
/// it through [`Self::buffer`], the ANE through a [`Binding`].
pub struct Surface {
    owner: Arc<SurfaceRef>,
    buffer: MetalBuffer,
    layout: SurfaceLayout,
}

impl Surface {
    #[expect(unsafe_code)]
    pub fn new(context: &MetalContext, layout: SurfaceLayout) -> crate::Result<Self> {
        let pixel = if layout.fp16 { PIXEL_FP16 } else { PIXEL_INT8 };
        let values = [
            NSNumber::new_u32(layout.width),
            NSNumber::new_u32(layout.rows),
            NSNumber::new_u32(layout.element_bytes() as u32),
            NSNumber::new_usize(layout.stride),
            NSNumber::new_usize(layout.bytes),
            NSNumber::new_u32(pixel),
        ];
        // SAFETY: the IOSurface property keys are CFString constants, toll-free
        // bridged to NSString.
        let keys: [&NSString; 6] = unsafe {
            [
                kIOSurfaceWidth,
                kIOSurfaceHeight,
                kIOSurfaceBytesPerElement,
                kIOSurfaceBytesPerRow,
                kIOSurfaceAllocSize,
                kIOSurfacePixelFormat,
            ]
            .map(|key: &CFString| &*std::ptr::from_ref(key).cast::<NSString>())
        };
        let refs = values.iter().map(|value| &**value).collect::<Vec<_>>();
        let properties = NSDictionary::from_slices(&keys, &refs);
        // SAFETY: NSDictionary is toll-free bridged to CFDictionary, and the
        // properties have the types IOSurfaceCreate documents.
        let surface =
            unsafe { IOSurfaceRef::new(&*Retained::as_ptr(&properties).cast::<CFDictionary>()) }
                .ok_or_else(|| {
                    ane_error(format!(
                        "IOSurfaceCreate {}x{} failed",
                        layout.rows, layout.width
                    ))
                })?;
        if surface.bytes_per_row() != layout.stride || surface.alloc_size() < layout.bytes {
            return Err(ane_error("IOSurface took another row stride or size"));
        }
        let base = surface.base_address().cast::<u8>();
        let surface = Arc::new(SurfaceRef(surface));
        let owner: Arc<dyn Send + Sync> = surface.clone();
        // SAFETY: the surface's memory is page-aligned and `layout.bytes`
        // (whole pages) long, and `owner` keeps the surface alive as long as
        // the Metal buffer exists.
        let buffer = unsafe {
            MetalBuffer::from_bytes_no_copy(context.device(), base, layout.bytes, owner)?
        };
        Ok(Self {
            owner: surface,
            buffer,
            layout,
        })
    }

    #[must_use]
    pub const fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }

    #[must_use]
    pub const fn layout(&self) -> SurfaceLayout {
        self.layout
    }

    fn raw(&self) -> &IOSurfaceRef {
        &self.owner.0
    }
}

struct Procedure {
    name: String,
    inputs: Vec<String>,
    input_symbols: Vec<u32>,
    output_symbol: u32,
}

struct ProgramInner {
    model: Shared<AnyObject>,
}

impl Drop for ProgramInner {
    #[expect(unsafe_code)]
    fn drop(&mut self) {
        let model = Shared(self.model.0.clone());
        // A daemon that never answers must not hang the process at exit.
        let _ = on_service_thread("ANE unload", Duration::from_secs(10), move || {
            let model = model;
            let api = api()?;
            catch("ANE unload", || {
                // SAFETY: the selector's encoding was checked at resolution.
                let _: Result<(), Retained<NSError>> = unsafe {
                    msg_send![&*api.client.0, unloadModel: &*model.0, options: &*NSDictionary::<NSString, AnyObject>::new(), qos: QOS, error: _]
                };
            })
        });
    }
}

/// Serialize every compile, load and unload (Kapitsa runs them on one serial
/// queue), waiting at most `limit`; a call that outlives it keeps running on
/// its own thread and only touches shared state.
fn on_service_thread<R: Send + 'static>(
    what: &'static str,
    limit: Duration,
    body: impl FnOnce() -> crate::Result<R> + Send + 'static,
) -> crate::Result<R> {
    static SERVICE: Mutex<()> = Mutex::new(());
    let (sender, receiver) = mpsc::channel();
    std::thread::Builder::new()
        .name("ane-service".into())
        .spawn(move || {
            let _serial = lock(&SERVICE);
            let result = autoreleasepool(|_| body());
            let _ = sender.send(result);
        })
        .map_err(|error| ane_error(format!("{what}: {error}")))?;
    receiver.recv_timeout(limit).map_err(|_| {
        ane_error(format!(
            "{what} did not finish within {} s",
            limit.as_secs()
        ))
    })?
}

/// Write `bytes` to `path` unless it already holds exactly them.
fn write_if_changed(path: &Path, bytes: &[u8]) -> crate::Result<()> {
    static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    let write = WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = path.with_extension(format!("tmp-{}-{write}", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

/// What [`Program::compile`] did.
#[derive(Clone, Debug)]
pub struct ProgramReport {
    /// The daemon still held a compilation under this program's hash.
    pub cached: bool,
    /// A held compilation failed to load and was purged and compiled again.
    pub recompiled: bool,
    pub compile: Duration,
    pub load: Duration,
    pub directory: PathBuf,
}

/// A compiled and loaded MIL program (one or more functions, the daemon's
/// procedures).
pub struct Program {
    inner: Arc<ProgramInner>,
    procedures: Arc<Vec<Procedure>>,
    report: ProgramReport,
}

struct Loaded {
    model: Shared<AnyObject>,
    procedures: Vec<Procedure>,
    cached: bool,
    recompiled: bool,
    compile: Duration,
    load: Duration,
}

#[expect(unsafe_code)]
fn object_for_key(dictionary: &AnyObject, key: &str) -> Option<Retained<AnyObject>> {
    let key = NSString::from_str(key);
    // SAFETY: `dictionary` is an NSDictionary from the model description.
    unsafe { msg_send![dictionary, objectForKey: &*key] }
}

#[expect(unsafe_code)]
fn array_items(array: &AnyObject) -> Vec<Retained<AnyObject>> {
    // SAFETY: `array` is an NSArray from the model description.
    let count: usize = unsafe { msg_send![array, count] };
    (0..count)
        .filter_map(|index| {
            // SAFETY: as above, `index < count`.
            let item: Option<Retained<AnyObject>> =
                unsafe { msg_send![array, objectAtIndex: index] };
            item
        })
        .collect()
}

#[expect(unsafe_code)]
fn number(object: &AnyObject) -> u32 {
    // SAFETY: the description's indices are NSNumbers.
    unsafe { msg_send![object, unsignedIntValue] }
}

#[expect(unsafe_code)]
fn string(object: &AnyObject) -> String {
    // SAFETY: the description's names are NSStrings; UTF8String is valid
    // while `object` lives.
    unsafe {
        let raw: *const c_char = msg_send![object, UTF8String];
        if raw.is_null() {
            String::new()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }
}

#[expect(unsafe_code)]
fn describe(model: &AnyObject) -> crate::Result<Vec<Procedure>> {
    let missing = || ane_error("the ANE model does not describe its procedures");
    // SAFETY: the selector's encoding was checked at resolution.
    let attributes: Option<Retained<AnyObject>> = unsafe { msg_send![model, modelAttributes] };
    let attributes = attributes.ok_or_else(missing)?;
    let description = object_for_key(&attributes, "ANEFModelDescription").ok_or_else(missing)?;
    let symbols = object_for_key(&description, "kANEFModelInputSymbolsArrayKey")
        .map(|array| array_items(&array))
        .ok_or_else(missing)?;
    let names =
        object_for_key(&description, "kANEFModelProcedureNameToIDMapKey").ok_or_else(missing)?;
    let entries = object_for_key(&description, "ANEFModelProcedures")
        .map(|array| array_items(&array))
        .ok_or_else(missing)?;
    let mut procedures = (0..entries.len())
        .map(|_| Procedure {
            name: String::new(),
            inputs: Vec::new(),
            input_symbols: Vec::new(),
            output_symbol: 0,
        })
        .collect::<Vec<_>>();
    // SAFETY: `names` is an NSDictionary of NSString to NSNumber.
    let keys: Option<Retained<AnyObject>> = unsafe { msg_send![&*names, allKeys] };
    for key in keys.map(|keys| array_items(&keys)).unwrap_or_default() {
        let id = object_for_key(&names, &string(&key)).map(|id| number(&id));
        let procedure = id
            .and_then(|id| procedures.get_mut(id as usize))
            .ok_or_else(|| ane_error("ANE procedure id out of range"))?;
        procedure.name = string(&key);
    }
    for entry in entries {
        let id = object_for_key(&entry, "ANEFModelProcedureID")
            .map(|id| number(&id))
            .ok_or_else(missing)?;
        let procedure = procedures
            .get_mut(id as usize)
            .ok_or_else(|| ane_error("ANE procedure id out of range"))?;
        let outputs = object_for_key(&entry, "ANEFModelOutputSymbolIndexArray")
            .map(|array| array_items(&array))
            .unwrap_or_default();
        let [output] = outputs.as_slice() else {
            return Err(ane_error("ANE procedure with other than one output"));
        };
        procedure.output_symbol = number(output);
        for symbol in object_for_key(&entry, "ANEFModelInputSymbolIndexArray")
            .map(|array| array_items(&array))
            .unwrap_or_default()
        {
            let index = number(&symbol);
            let name = symbols
                .get(index as usize)
                .ok_or_else(|| ane_error("ANE input symbol out of range"))?;
            procedure.input_symbols.push(index);
            procedure.inputs.push(string(name));
        }
    }
    Ok(procedures)
}

#[expect(unsafe_code)]
fn compile_and_load(directory: &Path, key: &str) -> crate::Result<Loaded> {
    let api = api()?;
    let url =
        NSURL::fileURLWithPath_isDirectory(&NSString::from_str(&directory.to_string_lossy()), true);
    let key = NSString::from_str(key);
    let model = catch("_ANEModel modelAtURL:key:", || {
        // SAFETY: the selector's encoding was checked at resolution.
        let model: Option<Retained<AnyObject>> =
            unsafe { msg_send![api.model, modelAtURL: &*url, key: &*key] };
        model
    })?
    .ok_or_else(|| ane_error("_ANEModel modelAtURL:key: returned nil"))?;
    let options = NSDictionary::from_slices(
        &[
            &*NSString::from_str("kANEFModelType"),
            &*NSString::from_str("kANEFNetPlistFilenameKey"),
        ],
        &[
            &*NSString::from_str("kANEFModelMIL"),
            &*NSString::from_str("model.mil"),
        ],
    );
    let empty = NSDictionary::<NSString, AnyObject>::new();
    let client = &*api.client.0;
    let compile = || -> crate::Result<()> {
        catch("ANE compile", || {
            // SAFETY: the selector's encoding was checked at resolution.
            let result: Result<(), Retained<NSError>> = unsafe {
                msg_send![client, compileModel: &*model, options: &*options, qos: QOS, error: _]
            };
            result
        })?
        .map_err(|error| ane_error(format!("ANE compile failed: {}", error_text(&error))))
    };
    let load = || -> crate::Result<Result<(), String>> {
        catch("ANE load", || {
            // SAFETY: the selector's encoding was checked at resolution.
            let result: Result<(), Retained<NSError>> = unsafe {
                msg_send![client, loadModel: &*model, options: &*empty, qos: QOS, error: _]
            };
            result.map_err(|error| error_text(&error))
        })
    };
    let started = Instant::now();
    let cached = catch("ANE compiledModelExistsFor", || {
        // SAFETY: the selector's encoding was checked at resolution.
        let exists: bool = unsafe { msg_send![client, compiledModelExistsFor: &*model] };
        exists
    })?;
    if !cached {
        compile()?;
    }
    let compile_time = started.elapsed();
    let started = Instant::now();
    let recompiled = if let Err(reason) = load()? {
        // The daemon may hold a compilation it cannot load (a stale cache
        // after an OS update): purge, compile again, load again.
        eprintln!("ane: load failed ({reason}); purging and recompiling");
        catch("ANE purge", || {
            // SAFETY: the selector's encoding was checked at resolution.
            let (): () = unsafe { msg_send![client, purgeCompiledModel: &*model] };
        })?;
        compile()?;
        load()?.map_err(|reason| ane_error(format!("ANE load failed: {reason}")))?;
        true
    } else {
        false
    };
    let load_time = started.elapsed();
    let procedures = catch("ANE modelAttributes", || describe(&model))??;
    Ok(Loaded {
        model: Shared(model),
        procedures,
        cached: cached && !recompiled,
        recompiled,
        compile: compile_time,
        load: load_time,
    })
}

impl Program {
    /// Write `model.mil` and `weights.bin` under
    /// `$(DARWIN_USER_CACHE_DIR)/<tag>/<hash>/`, compile unless the daemon
    /// still holds a compilation under that hash, and load, waiting at most
    /// `limit`.
    pub fn compile(mil: &str, blob: &[u8], tag: &str, limit: Duration) -> crate::Result<Self> {
        api()?;
        let key = format!("{:016x}", fnv1a(blob, fnv1a(mil.as_bytes(), FNV_OFFSET)));
        let directory = user_cache_dir()?.join(tag).join(&key);
        std::fs::create_dir_all(&directory)?;
        write_if_changed(&directory.join("model.mil"), mil.as_bytes())?;
        write_if_changed(&directory.join("weights.bin"), blob)?;
        let path = directory.clone();
        let loaded = on_service_thread("ANE compile/load", limit, move || {
            compile_and_load(&path, &key)
        })?;
        Ok(Self {
            inner: Arc::new(ProgramInner {
                model: loaded.model,
            }),
            procedures: Arc::new(loaded.procedures),
            report: ProgramReport {
                cached: loaded.cached,
                recompiled: loaded.recompiled,
                compile: loaded.compile,
                load: loaded.load,
                directory,
            },
        })
    }

    #[must_use]
    pub const fn report(&self) -> &ProgramReport {
        &self.report
    }

    /// Index of the function named `name`.
    #[must_use]
    pub fn procedure(&self, name: &str) -> Option<usize> {
        self.procedures
            .iter()
            .position(|procedure| procedure.name == name)
    }

    /// Input names of procedure `index`, in the order [`Self::bind`] takes.
    #[must_use]
    pub fn inputs(&self, index: usize) -> &[String] {
        self.procedures
            .get(index)
            .map_or(&[], |procedure| procedure.inputs.as_slice())
    }

    /// Procedure `index` bound to fixed surfaces: `inputs` in
    /// [`Self::inputs`] order.
    #[expect(unsafe_code)]
    pub fn bind(
        &self,
        index: usize,
        inputs: &[&Surface],
        output: &Surface,
    ) -> crate::Result<Binding> {
        let api = api()?;
        let procedure = self
            .procedures
            .get(index)
            .ok_or_else(|| ane_error("ANE bind: no such procedure"))?;
        if inputs.len() != procedure.inputs.len() {
            return Err(ane_error("ANE bind: input count mismatch"));
        }
        autoreleasepool(|_| {
            let wrap = |surface: &Surface| -> crate::Result<Retained<AnyObject>> {
                catch("_ANEIOSurfaceObject objectWithIOSurface:", || {
                    // SAFETY: the selector's encoding was checked at resolution.
                    let object: Option<Retained<AnyObject>> =
                        unsafe { msg_send![api.surface, objectWithIOSurface: surface.raw()] };
                    object
                })?
                .ok_or_else(|| ane_error("_ANEIOSurfaceObject returned nil"))
            };
            let objects = inputs
                .iter()
                .map(|surface| wrap(surface))
                .collect::<crate::Result<Vec<_>>>()?;
            let indices = procedure
                .input_symbols
                .iter()
                .map(|&symbol| NSNumber::new_u32(symbol))
                .collect::<Vec<_>>();
            Ok(Binding {
                inputs: Shared(NSArray::from_retained_slice(&objects)),
                input_indices: Shared(NSArray::from_retained_slice(&indices)),
                outputs: Shared(NSArray::from_retained_slice(&[wrap(output)?])),
                output_indices: Shared(NSArray::from_retained_slice(&[NSNumber::new_u32(
                    procedure.output_symbol,
                )])),
                procedure: Shared(NSNumber::new_usize(index)),
                program: Arc::clone(&self.inner),
            })
        })
    }
}

/// A procedure bound to fixed surfaces; keeps its program loaded.
pub struct Binding {
    inputs: Shared<NSArray<AnyObject>>,
    input_indices: Shared<NSArray<NSNumber>>,
    outputs: Shared<NSArray<AnyObject>>,
    output_indices: Shared<NSArray<NSNumber>>,
    procedure: Shared<NSNumber>,
    program: Arc<ProgramInner>,
}

impl Binding {
    #[expect(unsafe_code)]
    fn request(&self, api: &Api, events: Option<&AnyObject>) -> crate::Result<Retained<AnyObject>> {
        let none: Option<&AnyObject> = None;
        catch("_ANERequest creation", || {
            // SAFETY: the selector's encoding was checked at resolution.
            let request: Option<Retained<AnyObject>> = unsafe {
                msg_send![
                    api.request,
                    requestWithInputs: &*self.inputs.0,
                    inputIndices: &*self.input_indices.0,
                    outputs: &*self.outputs.0,
                    outputIndices: &*self.output_indices.0,
                    weightsBuffer: none,
                    perfStats: none,
                    procedureIndex: &*self.procedure.0,
                    sharedEvents: events,
                    transactionHandle: none
                ]
            };
            request
        })?
        .ok_or_else(|| ane_error("ANE request creation failed"))
    }

    /// Synchronous evaluation (no events), returning its wall time. It has no
    /// timeout: callers that cannot wait forever run it on a thread of their
    /// own.
    #[expect(unsafe_code)]
    pub fn evaluate(&self) -> crate::Result<Duration> {
        let api = api()?;
        autoreleasepool(|_| {
            let request = self.request(api, None)?;
            let empty = NSDictionary::<NSString, AnyObject>::new();
            let started = Instant::now();
            catch("ANE evaluate", || {
                // SAFETY: the selector's encoding was checked at resolution.
                let result: Result<(), Retained<NSError>> = unsafe {
                    msg_send![&*api.client.0, evaluateWithModel: &*self.program.model.0, options: &*empty, request: &*request, qos: QOS, error: _]
                };
                result
            })?
            .map_err(|error| ane_error(format!("ANE evaluate failed: {}", error_text(&error))))?;
            Ok(started.elapsed())
        })
    }

    /// Asynchronous evaluation: it starts once `wait_event` reaches `wait`
    /// and raises `signal_event` to `signal` when done; `done(ok)` reports
    /// from a daemon callback thread.
    #[expect(unsafe_code)]
    fn enqueue(
        &self,
        wait_event: &ProtocolObject<dyn MTLSharedEvent>,
        wait: u64,
        signal_event: &ProtocolObject<dyn MTLSharedEvent>,
        signal: u64,
        done: Box<dyn FnOnce(bool) + Send>,
    ) -> crate::Result<()> {
        let api = api()?;
        let wait_object: &AnyObject = wait_event.as_ref();
        let signal_object: &AnyObject = signal_event.as_ref();
        // The daemon shares events by their Mach ports; an event wrapped by
        // Metal's API validation layer lacks one.
        let port = sel!(eventPort);
        if !wait_object.class().responds_to(port) || !signal_object.class().responds_to(port) {
            return Err(ane_error(
                "the Metal shared event has no eventPort (Metal API validation on?)",
            ));
        }
        autoreleasepool(|_| {
            let signal_entry = catch("_ANESharedSignalEvent", || {
                // SAFETY: the selector's encoding was checked at resolution.
                let entry: Option<Retained<AnyObject>> = unsafe {
                    msg_send![api.signal_event, signalEventWithValue: signal, symbolIndex: 0_u32, eventType: 0_i64, sharedEvent: signal_object]
                };
                entry
            })?
            .ok_or_else(|| ane_error("ANE signal event creation failed"))?;
            let wait_entry = catch("_ANESharedWaitEvent", || {
                // SAFETY: the selector's encoding was checked at resolution.
                let entry: Option<Retained<AnyObject>> = unsafe {
                    msg_send![api.wait_event, waitEventWithValue: wait, sharedEvent: wait_object, eventType: 0_u64]
                };
                entry
            })?
            .ok_or_else(|| ane_error("ANE wait event creation failed"))?;
            let signals = NSArray::from_retained_slice(&[signal_entry]);
            let waits = NSArray::from_retained_slice(&[wait_entry]);
            let events = catch("_ANESharedEvents", || {
                // SAFETY: the selector's encoding was checked at resolution.
                let events: Option<Retained<AnyObject>> = unsafe {
                    msg_send![api.events, sharedEventsWithSignalEvents: &*signals, waitEvents: &*waits]
                };
                events
            })?
            .ok_or_else(|| ane_error("ANE shared events creation failed"))?;
            let request = self.request(api, Some(&events))?;
            // With shared events the evaluation is asynchronous and needs a
            // completion handler.
            let done = Mutex::new(Some(done));
            let handler = RcBlock::new(move |success: Bool, _error: *mut AnyObject| {
                let once = lock(&done).take();
                if let Some(done) = once {
                    done(success.as_bool());
                }
            });
            catch("ANE setCompletionHandler:", || {
                // SAFETY: the selector's encoding was checked at resolution;
                // the request copies the block.
                let (): () = unsafe { msg_send![&*request, setCompletionHandler: &*handler] };
            })?;
            let empty = NSDictionary::<NSString, AnyObject>::new();
            catch("ANE evaluate (async)", || {
                // SAFETY: the selector's encoding was checked at resolution.
                let result: Result<(), Retained<NSError>> = unsafe {
                    msg_send![&*api.client.0, evaluateWithModel: &*self.program.model.0, options: &*empty, request: &*request, qos: QOS, error: _]
                };
                result
            })?
            .map_err(|error| {
                ane_error(format!("ANE evaluate (async) failed: {}", error_text(&error)))
            })
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Started(Instant),
    Succeeded,
    Failed,
}

struct HandoffInner {
    /// Highest values handed out.
    ready_value: u64,
    done_value: u64,
    jobs: HashMap<u64, Phase>,
    next_job: u64,
    outstanding: usize,
    retired: Option<String>,
    stop: bool,
    /// When the work [`Handoff::arm`]ed must be finished.
    deadline: Option<Instant>,
}

struct HandoffState {
    ready: Shared<ProtocolObject<dyn MTLSharedEvent>>,
    done: Shared<ProtocolObject<dyn MTLSharedEvent>>,
    listener: Shared<MTLSharedEventListener>,
    bound: Duration,
    inner: Mutex<HandoffInner>,
    changed: Condvar,
}

impl HandoffState {
    /// Give up on the ANE: raise both events to the highest values handed
    /// out, so every pending GPU wait passes.
    fn retire_locked(&self, inner: &mut HandoffInner, why: &str) {
        if inner.retired.is_none() {
            inner.retired = Some(why.to_owned());
        }
        if self.ready.0.signaledValue() < inner.ready_value {
            self.ready.0.setSignaledValue(inner.ready_value);
        }
        if self.done.0.signaledValue() < inner.done_value {
            self.done.0.setSignaledValue(inner.done_value);
        }
        self.changed.notify_all();
    }

    fn report(&self, job: u64, ok: bool) {
        let mut inner = lock(&self.inner);
        inner.outstanding = inner.outstanding.saturating_sub(1);
        let phase = inner.jobs.get(&job).copied();
        if matches!(phase, Some(Phase::Queued | Phase::Started(_))) {
            if ok {
                inner.jobs.insert(job, Phase::Succeeded);
            } else {
                inner.jobs.insert(job, Phase::Failed);
                self.retire_locked(&mut inner, "a Neural Engine evaluation failed");
            }
        }
        drop(inner);
        self.changed.notify_all();
    }

    /// The watchdog: wakes on job starts and completions, and at the earliest
    /// deadline of a started job; a job past its deadline retires the
    /// handoff.
    fn watch(&self) {
        let mut inner = lock(&self.inner);
        loop {
            if inner.stop {
                return;
            }
            let now = Instant::now();
            let mut earliest: Option<Instant> = None;
            let mut expired = false;
            for phase in inner.jobs.values_mut() {
                if let Phase::Started(started) = *phase {
                    let deadline = started + self.bound;
                    if deadline <= now {
                        *phase = Phase::Failed;
                        expired = true;
                    } else {
                        earliest = Some(earliest.map_or(deadline, |e| e.min(deadline)));
                    }
                }
            }
            if expired {
                let why = format!(
                    "a Neural Engine evaluation did not complete within {} ms",
                    self.bound.as_millis()
                );
                self.retire_locked(&mut inner, &why);
            }
            match inner.deadline {
                Some(deadline) if deadline <= now => {
                    inner.deadline = None;
                    self.retire_locked(&mut inner, "a Neural Engine chunk passed its deadline");
                }
                Some(deadline) => earliest = Some(earliest.map_or(deadline, |e| e.min(deadline))),
                None => {}
            }
            inner = match earliest {
                Some(deadline) => {
                    self.changed
                        .wait_timeout(inner, deadline.saturating_duration_since(now))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
                None => self
                    .changed
                    .wait(inner)
                    .unwrap_or_else(PoisonError::into_inner),
            };
        }
    }
}

/// GPU and ANE ordering through two shared events.
///
/// The GPU raises `ready` and the ANE raises `done`, each in its own
/// increasing order
/// ([`Self::next_ready`], [`Self::next_done`]), so jobs of two row halves can
/// interleave without a value ever going back. [`Self::queue`] registers an
/// ANE job that waits for `ready` and signals `done`; once the ready event
/// reaches its value a watchdog of `bound` starts. A job that fails, times
/// out or cannot start retires the handoff: both events are raised to the
/// highest values handed out, so every pending wait passes, and later values
/// are signalled at once.
pub struct Handoff {
    state: Arc<HandoffState>,
    watchdog: Option<std::thread::JoinHandle<()>>,
}

impl Handoff {
    pub fn new(context: &MetalContext, bound: Duration) -> crate::Result<Self> {
        let event = || {
            context
                .device()
                .newSharedEvent()
                .map(Shared)
                .ok_or_else(|| ane_error("newSharedEvent returned nil"))
        };
        let ready = event()?;
        let done = event()?;
        let inner = HandoffInner {
            ready_value: ready.0.signaledValue(),
            done_value: done.0.signaledValue(),
            jobs: HashMap::new(),
            next_job: 0,
            outstanding: 0,
            retired: None,
            stop: false,
            deadline: None,
        };
        let state = Arc::new(HandoffState {
            ready,
            done,
            listener: Shared(MTLSharedEventListener::sharedListener()),
            bound,
            inner: Mutex::new(inner),
            changed: Condvar::new(),
        });
        let watched = Arc::clone(&state);
        let watchdog = std::thread::Builder::new()
            .name("ane-watchdog".into())
            .spawn(move || watched.watch())
            .map_err(|error| ane_error(format!("watchdog thread: {error}")))?;
        Ok(Self {
            state,
            watchdog: Some(watchdog),
        })
    }

    /// The GPU-to-ANE event.
    #[must_use]
    pub fn ready_event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.state.ready.0
    }

    /// The ANE-to-GPU event.
    #[must_use]
    pub fn done_event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.state.done.0
    }

    #[must_use]
    pub fn next_ready(&self) -> u64 {
        let mut inner = lock(&self.state.inner);
        inner.ready_value += 1;
        if inner.retired.is_some() && self.state.ready.0.signaledValue() < inner.ready_value {
            self.state.ready.0.setSignaledValue(inner.ready_value);
        }
        inner.ready_value
    }

    #[must_use]
    pub fn next_done(&self) -> u64 {
        let mut inner = lock(&self.state.inner);
        inner.done_value += 1;
        if inner.retired.is_some() && self.state.done.0.signaledValue() < inner.done_value {
            self.state.done.0.setSignaledValue(inner.done_value);
        }
        inner.done_value
    }

    /// Enqueue `binding` on the ANE to wait for `ready` and signal `done`.
    /// After retirement, or when it cannot start, nothing is enqueued and the
    /// GPU's waits pass at once. `skip` registers the job without starting it
    /// (a test of the watchdog).
    #[expect(unsafe_code)]
    pub fn queue(&self, binding: &Binding, ready: u64, done: u64, skip: bool) {
        let state = &self.state;
        let job = {
            let mut inner = lock(&state.inner);
            inner.next_job += 1;
            let job = inner.next_job;
            if inner.retired.is_some() {
                inner.jobs.insert(job, Phase::Failed);
                return;
            }
            inner.jobs.insert(job, Phase::Queued);
            inner.outstanding += 1;
            job
        };
        let reporter = Arc::clone(state);
        let started = if skip {
            Ok(())
        } else {
            binding.enqueue(
                &state.ready.0,
                ready,
                &state.done.0,
                done,
                Box::new(move |ok| reporter.report(job, ok)),
            )
        };
        if let Err(error) = started {
            let mut inner = lock(&state.inner);
            inner.outstanding = inner.outstanding.saturating_sub(1);
            inner.jobs.insert(job, Phase::Failed);
            let why = format!("a Neural Engine evaluation could not start: {error}");
            state.retire_locked(&mut inner, &why);
            drop(inner);
            return;
        }
        let watched = Arc::clone(state);
        let notify = RcBlock::new(
            move |_event: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _value: u64| {
                let mut inner = lock(&watched.inner);
                if let Some(phase @ Phase::Queued) = inner.jobs.get_mut(&job) {
                    *phase = Phase::Started(Instant::now());
                    watched.changed.notify_all();
                }
            },
        );
        // SAFETY: the listener copies the block, which owns only Send + Sync
        // state.
        unsafe {
            state.ready.0.notifyListener_atValue_block(
                &state.listener.0,
                ready,
                RcBlock::as_ptr(&notify),
            );
        }
    }

    /// Retire unless [`Self::finish`] is called within `limit`: the last
    /// resort for an ANE job whose ready event never reports, which no
    /// per-job watchdog would see start.
    pub fn arm(&self, limit: Duration) {
        lock(&self.state.inner).deadline = Some(Instant::now() + limit);
        self.state.changed.notify_all();
    }

    /// Wait (at most one bound) for the jobs queued since the last finish.
    /// `false` if any failed or did not report. Disarms [`Self::arm`].
    pub fn finish(&self) -> bool {
        let state = &self.state;
        let mut inner = lock(&state.inner);
        inner.deadline = None;
        let settled = |inner: &HandoffInner| {
            inner.jobs.values().any(|phase| *phase == Phase::Failed)
                || inner.jobs.values().all(|phase| *phase == Phase::Succeeded)
        };
        let deadline = Instant::now() + state.bound;
        while !settled(&inner) {
            let now = Instant::now();
            if now >= deadline {
                let why = format!(
                    "a Neural Engine evaluation did not report within {} ms of its command",
                    state.bound.as_millis()
                );
                state.retire_locked(&mut inner, &why);
                break;
            }
            inner = state
                .changed
                .wait_timeout(inner, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let ok =
            inner.retired.is_none() && inner.jobs.values().all(|phase| *phase == Phase::Succeeded);
        inner.jobs.clear();
        ok
    }

    pub fn retire(&self, why: &str) {
        let mut inner = lock(&self.state.inner);
        self.state.retire_locked(&mut inner, why);
    }

    /// Why the handoff was retired, if it was.
    #[must_use]
    pub fn retired(&self) -> Option<String> {
        lock(&self.state.inner).retired.clone()
    }

    /// Evaluations enqueued that have not reported.
    #[must_use]
    pub fn outstanding(&self) -> usize {
        lock(&self.state.inner).outstanding
    }
}

impl Drop for Handoff {
    fn drop(&mut self) {
        let state = &self.state;
        let mut inner = lock(&state.inner);
        state.retire_locked(&mut inner, "the handoff ended");
        // Evaluations still in flight write surfaces their owner frees next;
        // give them one bound.
        let deadline = Instant::now() + state.bound;
        while inner.outstanding > 0 {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            inner = state
                .changed
                .wait_timeout(inner, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        inner.stop = true;
        state.changed.notify_all();
        drop(inner);
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
    }
}

#[cfg(test)]
mod tests;
