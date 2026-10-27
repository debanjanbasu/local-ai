//! Name resolution for web fetch: checked address lists, cached briefly, and
//! DNS HTTPS records (RFC 9460) that say a host serves HTTP/3.
//!
//! - Addresses come from the system resolver (`getaddrinfo`). A name is
//!   refused outright when any of its addresses is not permitted (public,
//!   see [`super::is_public`]), and only a fully checked list is cached, for
//!   [`ADDRESS_TTL`] (`getaddrinfo` reports no TTL, so this is a short,
//!   fixed bound well under common record TTLs). Every connection goes to
//!   an address from such a list; nothing resolves the name again between
//!   the check and the connection.
//! - The HTTPS record is asked of the system's DNS service
//!   (`DNSServiceQueryRecord`, so the same resolver configuration as
//!   `getaddrinfo`), only for its `alpn` and `port`: an alternative endpoint
//!   with another target name, and its address hints, are ignored, so the
//!   record can never add an address or a host.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::ToolError;

/// How long a checked address list is reused.
pub const ADDRESS_TTL: Duration = Duration::from_secs(30);
/// Names whose addresses are remembered.
const MAX_NAMES: usize = 256;

/// A checked address list and when it expires.
type Cached = (Arc<[IpAddr]>, Instant);

/// Checked addresses per name.
#[derive(Debug, Default)]
pub struct Resolver {
    cache: Mutex<HashMap<String, Cached>>,
    /// Test-only fixed answers, so local servers can stand in for names.
    #[cfg(test)]
    pub overrides: HashMap<String, Vec<IpAddr>>,
}

impl Resolver {
    /// The addresses of `host` (a DNS name or an IP literal), all of which
    /// `permitted` accepts, or an error naming the first that it refuses.
    pub async fn resolve(
        &self,
        host: &str,
        permitted: impl Fn(IpAddr) -> bool,
    ) -> Result<Arc<[IpAddr]>, ToolError> {
        if let Ok(ip) = host.parse::<IpAddr>() {
            return check(host, vec![ip], &permitted);
        }
        let now = Instant::now();
        if let Some((addresses, _)) = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(host)
            .filter(|(_, expires)| *expires > now)
        {
            return Ok(Arc::clone(addresses));
        }
        #[cfg(test)]
        let fixed = self.overrides.get(host).cloned();
        #[cfg(not(test))]
        let fixed: Option<Vec<IpAddr>> = None;
        let found = match fixed {
            Some(fixed) => fixed,
            None => tokio::net::lookup_host((host, 0))
                .await
                .map_err(|error| ToolError::Failed(format!("cannot resolve {host}: {error}")))?
                .map(|address| address.ip())
                .collect(),
        };
        let addresses = check(host, found, &permitted)?;
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if cache.len() >= MAX_NAMES {
            cache.retain(|_, (_, expires)| *expires > now);
            if cache.len() >= MAX_NAMES {
                cache.clear();
            }
        }
        cache.insert(host.to_owned(), (Arc::clone(&addresses), now + ADDRESS_TTL));
        drop(cache);
        Ok(addresses)
    }
}

/// `addresses`, de-duplicated and ordered for connecting (families
/// alternating, first family first, RFC 8305), if all are permitted.
fn check(
    host: &str,
    mut addresses: Vec<IpAddr>,
    permitted: &impl Fn(IpAddr) -> bool,
) -> Result<Arc<[IpAddr]>, ToolError> {
    if let Some(refused) = addresses.iter().find(|ip| !permitted(**ip)) {
        return Err(ToolError::Denied(format!(
            "{host} resolves to the non-public address {refused}, which is never contacted"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    addresses.retain(|ip| seen.insert(*ip));
    if addresses.is_empty() {
        return Err(ToolError::Failed(format!("{host} has no addresses")));
    }
    let first_v6 = addresses.first().is_some_and(IpAddr::is_ipv6);
    let (mut first, mut second): (Vec<IpAddr>, Vec<IpAddr>) = addresses
        .into_iter()
        .partition(|ip| ip.is_ipv6() == first_v6);
    first.reverse();
    second.reverse();
    let mut ordered = Vec::with_capacity(first.len() + second.len());
    while let Some(ip) = first.pop() {
        ordered.push(ip);
        ordered.extend(second.pop());
    }
    ordered.extend(second.into_iter().rev());
    Ok(ordered.into())
}

/// What an HTTPS record says about HTTP/3 on its owner's host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Https {
    /// The port, when the record names one.
    pub port: Option<u16>,
    pub ttl: Duration,
}

/// Parse one HTTPS record's RDATA for `host`: `Some` when it is a service
/// record for the same host (target `.` or `host` itself) whose `alpn`
/// lists `h3` and whose mandatory keys are all understood.
#[must_use]
pub fn parse_https(rdata: &[u8], host: &str, ttl: Duration) -> Option<Https> {
    let mut cursor = rdata;
    let priority = take_u16(&mut cursor)?;
    if priority == 0 {
        // Alias mode: names another host; never followed.
        return None;
    }
    let mut target = String::new();
    loop {
        let (&length, rest) = cursor.split_first()?;
        cursor = rest;
        if length == 0 {
            break;
        }
        let label = cursor.get(..usize::from(length))?;
        cursor = cursor.get(usize::from(length)..)?;
        if !target.is_empty() {
            target.push('.');
        }
        target.push_str(std::str::from_utf8(label).ok()?);
    }
    if !(target.is_empty() || target.eq_ignore_ascii_case(host.trim_end_matches('.'))) {
        return None;
    }
    let (mut h3, mut port) = (false, None);
    while !cursor.is_empty() {
        let key = take_u16(&mut cursor)?;
        let length = usize::from(take_u16(&mut cursor)?);
        let mut value = cursor.get(..length)?;
        cursor = cursor.get(length..)?;
        match key {
            // mandatory: every listed key must be one this reader may act
            // without (alpn, no-default-alpn, port, and the address hints,
            // which are ignored). ECH (5) and unknown keys refuse.
            0 => {
                while !value.is_empty() {
                    if !matches!(take_u16(&mut value)?, 1 | 2 | 3 | 4 | 6) {
                        return None;
                    }
                }
            }
            1 => {
                while let Some((&length, rest)) = value.split_first() {
                    let id = rest.get(..usize::from(length))?;
                    value = rest.get(usize::from(length)..)?;
                    h3 |= id == b"h3";
                }
            }
            3 => port = Some(take_u16(&mut value)?).filter(|port| *port != 0),
            _ => {}
        }
    }
    h3.then_some(Https { port, ttl })
}

fn take_u16(cursor: &mut &[u8]) -> Option<u16> {
    let (bytes, rest) = cursor.split_first_chunk::<2>()?;
    *cursor = rest;
    Some(u16::from_be_bytes(*bytes))
}

/// Look up `host`'s HTTPS records through the system DNS service, waiting at
/// most `timeout` (blocking; run it off the async threads). The first that
/// [`parse_https`] accepts, by priority, or `None`.
#[must_use]
pub fn lookup_https(host: &str, timeout: Duration) -> Option<Https> {
    let mut records = dnssd::query(host, 65, timeout);
    records.sort_by_key(|(rdata, _)| {
        rdata
            .first_chunk::<2>()
            .map(|bytes| u16::from_be_bytes(*bytes))
    });
    records
        .iter()
        .find_map(|(rdata, ttl)| parse_https(rdata, host, Duration::from_secs(u64::from(*ttl))))
}

#[cfg(target_os = "macos")]
mod dnssd {
    //! The few `dns_sd.h` calls needed for one record query. They live in
    //! libSystem, which every macOS binary links.

    use std::cell::{Cell, RefCell};
    use std::ffi::{CString, c_char, c_int, c_void};
    use std::time::{Duration, Instant};

    type ServiceRef = *mut c_void;
    type Reply = unsafe extern "C" fn(
        ServiceRef,
        u32,
        u32,
        i32,
        *const c_char,
        u16,
        u16,
        u16,
        *const c_void,
        u32,
        *mut c_void,
    );

    const MORE_COMING: u32 = 0x1;
    const ADD: u32 = 0x2;
    /// Deliver negative answers (`kDNSServiceErr_NoSuchRecord`) instead of
    /// waiting for a record that does not exist.
    const RETURN_INTERMEDIATES: u32 = 0x1000;
    /// Let the daemon end the query on its own timeout too.
    const TIMEOUT: u32 = 0x10000;
    const CLASS_IN: u16 = 1;

    #[expect(unsafe_code, reason = "declarations of libSystem's DNS service API")]
    unsafe extern "C" {
        fn DNSServiceQueryRecord(
            service: *mut ServiceRef,
            flags: u32,
            interface: u32,
            name: *const c_char,
            rrtype: u16,
            rrclass: u16,
            reply: Reply,
            context: *mut c_void,
        ) -> i32;
        fn DNSServiceRefSockFD(service: ServiceRef) -> c_int;
        fn DNSServiceProcessResult(service: ServiceRef) -> i32;
        fn DNSServiceRefDeallocate(service: ServiceRef);
    }

    #[derive(Default)]
    struct Answers {
        rrtype: u16,
        records: RefCell<Vec<(Vec<u8>, u32)>>,
        done: Cell<bool>,
    }

    #[expect(unsafe_code, reason = "a DNS service callback")]
    unsafe extern "C" fn reply(
        _service: ServiceRef,
        flags: u32,
        _interface: u32,
        error: i32,
        _name: *const c_char,
        rrtype: u16,
        _rrclass: u16,
        length: u16,
        rdata: *const c_void,
        ttl: u32,
        context: *mut c_void,
    ) {
        // SAFETY: `context` is the `Answers` that `query` passed, alive and
        // only shared (never mutably borrowed) while the daemon's replies
        // are processed on this thread.
        let answers = unsafe { &*context.cast::<Answers>() };
        if error == 0 && rrtype == answers.rrtype && flags & ADD != 0 && !rdata.is_null() {
            // SAFETY: the daemon passes `length` readable bytes at `rdata`
            // for the duration of the callback.
            let data =
                unsafe { std::slice::from_raw_parts(rdata.cast::<u8>(), usize::from(length)) };
            answers.records.borrow_mut().push((data.to_vec(), ttl));
        }
        if flags & MORE_COMING == 0 && (error != 0 || rrtype == answers.rrtype) {
            answers.done.set(true);
        }
    }

    /// Ends the query when dropped.
    struct Service(ServiceRef);

    impl Drop for Service {
        fn drop(&mut self) {
            // SAFETY: a reference `DNSServiceQueryRecord` created, released
            // once.
            #[expect(unsafe_code, reason = "releases the DNS service query")]
            unsafe {
                DNSServiceRefDeallocate(self.0);
            }
        }
    }

    /// The `rrtype` records of `name` (RDATA and TTL), waiting at most
    /// `timeout`.
    pub fn query(name: &str, rrtype: u16, timeout: Duration) -> Vec<(Vec<u8>, u32)> {
        let Ok(name) = CString::new(name) else {
            return Vec::new();
        };
        let answers = Answers {
            rrtype,
            ..Answers::default()
        };
        let mut service: ServiceRef = std::ptr::null_mut();
        // SAFETY: valid out-pointer, NUL-terminated name, a callback of the
        // declared type and a context that outlives the query (`answers`
        // lives until after `Service` is dropped below).
        #[expect(unsafe_code, reason = "starts a DNS service query")]
        let started = unsafe {
            DNSServiceQueryRecord(
                &raw mut service,
                RETURN_INTERMEDIATES | TIMEOUT,
                0,
                name.as_ptr(),
                rrtype,
                CLASS_IN,
                reply,
                (&raw const answers).cast_mut().cast(),
            )
        };
        if started != 0 || service.is_null() {
            return Vec::new();
        }
        let service = Service(service);
        // SAFETY: a live query reference.
        #[expect(unsafe_code, reason = "reads the query's socket")]
        let fd = unsafe { DNSServiceRefSockFD(service.0) };
        let deadline = Instant::now() + timeout;
        while !answers.done.get() && fd >= 0 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let mut poll = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let millis = c_int::try_from(remaining.as_millis().max(1)).unwrap_or(c_int::MAX);
            // SAFETY: one valid `pollfd`.
            #[expect(unsafe_code, reason = "waits for the daemon's answer with a deadline")]
            let ready = unsafe { libc::poll(&raw mut poll, 1, millis) };
            if ready <= 0 {
                break;
            }
            // SAFETY: a live query reference whose socket is readable; the
            // callback runs on this thread, inside this call.
            #[expect(unsafe_code, reason = "delivers the daemon's answer to `reply`")]
            let processed = unsafe { DNSServiceProcessResult(service.0) };
            if processed != 0 {
                break;
            }
        }
        drop(service);
        answers.records.into_inner()
    }
}

#[cfg(not(target_os = "macos"))]
mod dnssd {
    pub fn query(_name: &str, _rrtype: u16, _timeout: std::time::Duration) -> Vec<(Vec<u8>, u32)> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use std::net::IpAddr;
    use std::time::Duration;

    use super::{Https, check, parse_https};

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(text.get(at..at + 2).expect("pair"), 16).expect("hex"))
            .collect()
    }

    #[test]
    fn https_records_say_h3_only_for_the_same_host() {
        let minute = Duration::from_secs(60);
        // cloudflare.com's record (dig TYPE65): priority 1, target ".",
        // alpn h3,h2, ipv4hint, ipv6hint.
        let cloudflare = hex(concat!(
            "0001000001000602683302683200040008681084e5681085e5000600",
            "20260647000000000000000000681084e52606470000000000000000",
            "00681085e5",
        ));
        assert_eq!(cloudflare.len(), 61);
        assert_eq!(
            parse_https(&cloudflare, "cloudflare.com", minute),
            Some(Https {
                port: None,
                ttl: minute
            })
        );
        // google.com's: alpn h2,h3.
        assert!(parse_https(&hex("00010000010006026832026833"), "google.com", minute).is_some());
        // Only h2: nothing to learn.
        assert_eq!(
            parse_https(&hex("00010000010003026832"), "a.test", minute),
            None
        );
        // Target is the host itself, with a port.
        let named = hex("0001016104746573740000010003026833000300021151");
        assert_eq!(
            parse_https(&named, "a.test", minute),
            Some(Https {
                port: Some(4433),
                ttl: minute
            })
        );
        // Another target host, alias mode, mandatory ECH, or truncation:
        // refused.
        let other = hex("0001046576696c04746573740000010003026833");
        assert_eq!(parse_https(&other, "a.test", minute), None);
        assert!(parse_https(&other, "evil.test", minute).is_some());
        assert_eq!(
            parse_https(&hex("00000161047465737400"), "a.test", minute),
            None
        );
        let mandatory_ech = hex("0001000000000200050001000302683300050000");
        assert_eq!(parse_https(&mandatory_ech, "a.test", minute), None);
        assert_eq!(
            parse_https(
                &cloudflare[..cloudflare.len() - 3],
                "cloudflare.com",
                minute
            ),
            None
        );
    }

    #[test]
    fn address_lists_are_all_checked_and_interleaved() {
        let ips: Vec<IpAddr> = ["2606:4700::1", "2606:4700::2", "104.16.1.1", "2606:4700::1"]
            .iter()
            .map(|ip| ip.parse().expect("ip"))
            .collect();
        let ordered = check("a.test", ips.clone(), &|_| true).expect("public");
        let expected: Vec<IpAddr> = ["2606:4700::1", "104.16.1.1", "2606:4700::2"]
            .iter()
            .map(|ip| ip.parse().expect("ip"))
            .collect();
        assert_eq!(&*ordered, expected.as_slice());
        let mut mixed = ips;
        mixed.push("10.0.0.1".parse().expect("ip"));
        let error = check("a.test", mixed, &|ip| super::super::is_public(ip)).expect_err("private");
        assert!(
            error.to_string().contains("non-public address 10.0.0.1"),
            "{error}"
        );
    }
}
