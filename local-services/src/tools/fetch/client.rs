//! The web fetch transport: one `GET` to a checked address over HTTP/3,
//! HTTP/2 or HTTP/1.1, on a pooled connection when one is live.
//!
//! - **Transport.** An `https` origin known to serve HTTP/3 (from `Alt-Svc`
//!   or a DNS HTTPS record, see [`super::altsvc`]) is tried over QUIC first,
//!   with TCP started after a short head start ([`Timeouts::head_start`]);
//!   whichever connects first is used. If TCP wins, the QUIC attempt
//!   finishes in the background and joins the pool for later requests; a
//!   QUIC failure marks the origin broken for a while (5 minutes, doubling
//!   per failure). Other origins use TCP, with ALPN `h2` then `http/1.1`.
//!   Within a transport, a host's addresses are tried in turn, each started
//!   [`STAGGER`] after the last (RFC 8305), first success wins.
//! - **TLS.** One rustls configuration (ring, the platform verifier), built
//!   once, is shared by both transports with per-transport ALPN and session
//!   caches, so a returning origin resumes its session. QUIC sends the
//!   request as 0-RTT early data when it holds a ticket: requests here are
//!   only bodiless `GET`s, idempotent by definition, so a replay changes
//!   nothing; a request lost to a rejected 0-RTT is retried once, on a full
//!   handshake.
//! - **Pool.** At most one multiplexed (HTTP/2 or HTTP/3) connection and
//!   [`MAX_IDLE_H1`] idle HTTP/1.1 connections per origin, for [`MAX_ORIGINS`]
//!   origins, each dropped after [`IDLE`] unused. An HTTP/1.1 connection is
//!   pooled again only after its body was read to the end; one dropped
//!   earlier (cut at a limit, cancelled) is closed, and a cancelled HTTP/2
//!   or HTTP/3 stream is reset. A request that fails on a pooled
//!   connection before any response is retried once on a new one.
//! - **Addresses.** Every connection, TCP or UDP, goes to an address from a
//!   resolution in which every address was checked ([`super::dns`]), and is
//!   checked again just before connecting.
//! - **Runtime.** Connections and their tasks live on one small runtime of
//!   their own, so a pooled connection outlives the runtime of the call that
//!   opened it (media fetches run on a fresh runtime each); a caller's
//!   deadline still bounds the whole exchange, and dropping the call
//!   abandons it.

use std::collections::HashMap;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use bytes::{Buf as _, Bytes};
use http_body_util::{BodyExt as _, Empty};
use hyper::client::conn::{http1, http2};
use hyper_util::rt::{TokioExecutor, TokioIo};
use rustls::pki_types::ServerName;

use super::{Target, ToolError, altsvc, dns, host_header};

/// Delay between connection attempts to successive addresses.
const STAGGER: Duration = Duration::from_millis(250);
/// How long an unused pooled connection is kept (QUIC's idle timeout too).
const IDLE: Duration = Duration::from_secs(30);
/// Idle HTTP/1.1 connections kept per origin.
const MAX_IDLE_H1: usize = 4;
/// Origins with pooled connections.
const MAX_ORIGINS: usize = 32;
/// TLS sessions remembered per transport, for resumption.
const SESSIONS: usize = 256;
/// Response header bytes accepted (HTTP/2 and HTTP/3; hyper bounds
/// HTTP/1.1's header buffer itself).
const MAX_HEADER_BYTES: u32 = 64 * 1024;
/// How long a DNS HTTPS lookup may take; the first fetch waits at most
/// [`HTTPS_RR_GRACE`] for it past the address lookup.
const HTTPS_RR_TIMEOUT: Duration = Duration::from_secs(1);
const HTTPS_RR_GRACE: Duration = Duration::from_millis(30);

/// Timing of the HTTP/3 race.
#[derive(Debug, Clone, Copy)]
pub struct Timeouts {
    /// QUIC's lead over TCP.
    pub head_start: Duration,
    /// The longest a QUIC handshake may take before it counts as failed.
    pub quic_handshake: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            head_start: Duration::from_millis(300),
            quic_handshake: Duration::from_secs(3),
        }
    }
}

/// The protocol a response came over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Http1,
    Http2,
    Http3,
}

/// An origin: what a pooled connection and an alternative belong to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key {
    pub https: bool,
    pub host: String,
    pub port: u16,
}

impl Key {
    #[must_use]
    pub fn new(https: bool, host: &str, port: u16) -> Self {
        Self {
            https,
            host: host.to_owned(),
            port,
        }
    }

    fn of(target: &Target) -> Self {
        Self::new(target.https, &target.host, target.port)
    }
}

type H3Sender = h3::client::SendRequest<h3_quinn::OpenStreams, Bytes>;
type H3Stream = h3::client::RequestStream<h3_quinn::BidiStream<Bytes>, Bytes>;

/// A connection to send a request on.
enum Sender {
    H1(http1::SendRequest<Empty<Bytes>>),
    H2(http2::SendRequest<Empty<Bytes>>),
    H3(H3Sender, quinn::Connection),
}

impl Sender {
    const fn protocol(&self) -> Protocol {
        match self {
            Self::H1(_) => Protocol::Http1,
            Self::H2(_) => Protocol::Http2,
            Self::H3(..) => Protocol::Http3,
        }
    }

    fn closed(&self) -> bool {
        match self {
            Self::H1(sender) => sender.is_closed(),
            Self::H2(sender) => sender.is_closed(),
            Self::H3(_, connection) => connection.close_reason().is_some(),
        }
    }

    /// Another handle on a multiplexed connection.
    fn share(&self) -> Option<Self> {
        match self {
            Self::H1(_) => None,
            Self::H2(sender) => Some(Self::H2(sender.clone())),
            Self::H3(sender, connection) => Some(Self::H3(sender.clone(), connection.clone())),
        }
    }
}

/// Live connections per origin.
#[derive(Default)]
struct Pool {
    origins: Mutex<HashMap<Key, Vec<(Sender, Instant)>>>,
}

impl Pool {
    fn origins(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Vec<(Sender, Instant)>>> {
        self.origins.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A live connection to `key`: a share of its multiplexed one, else an
    /// idle HTTP/1.1 one (taken out of the pool while in use).
    fn checkout(&self, key: &Key) -> Option<Sender> {
        let now = Instant::now();
        let mut origins = self.origins();
        origins.retain(|_, list| {
            list.retain(|(sender, since)| now.duration_since(*since) < IDLE && !sender.closed());
            !list.is_empty()
        });
        let list = origins.get_mut(key)?;
        let found = if let Some((sender, since)) = list
            .iter_mut()
            .rev()
            .find(|(sender, _)| !matches!(sender, Sender::H1(_)))
        {
            *since = now;
            sender.share()
        } else {
            list.pop().map(|(sender, _)| sender)
        };
        drop(origins);
        found
    }

    /// Pool `sender` for `key`. A multiplexed connection replaces the
    /// origin's previous one, which closes once its streams are done.
    fn put(&self, key: &Key, sender: Sender) {
        let now = Instant::now();
        let mut origins = self.origins();
        if !origins.contains_key(key) && origins.len() >= MAX_ORIGINS {
            let oldest = origins
                .iter()
                .min_by_key(|(_, list)| list.iter().map(|(_, since)| *since).max())
                .map(|(key, _)| key.clone());
            if let Some(oldest) = oldest {
                origins.remove(&oldest);
            }
        }
        let list = origins.entry(key.clone()).or_default();
        let idle_h1 = list
            .iter()
            .filter(|(sender, _)| matches!(sender, Sender::H1(_)))
            .count();
        if !matches!(sender, Sender::H1(_)) {
            list.retain(|(sender, _)| matches!(sender, Sender::H1(_)));
            list.push((sender, now));
        } else if idle_h1 < MAX_IDLE_H1 {
            list.push((sender, now));
        }
        drop(origins);
    }
}

/// A response whose body has not been read yet.
pub struct Response {
    pub status: http::StatusCode,
    pub headers: http::HeaderMap,
    /// Test-only: the protocol the response came over.
    #[cfg(test)]
    pub protocol: Protocol,
    body: Body,
}

enum Body {
    Hyper(hyper::body::Incoming, Option<GiveBack>),
    H3(Box<H3Stream>),
}

/// An HTTP/1.1 connection to pool again once its body has ended.
struct GiveBack {
    pool: Arc<Pool>,
    key: Key,
    sender: http1::SendRequest<Empty<Bytes>>,
}

impl Response {
    /// The next piece of the body, or `None` at its end.
    pub async fn chunk(&mut self) -> Result<Option<Bytes>, ToolError> {
        match &mut self.body {
            Body::Hyper(incoming, give_back) => loop {
                match incoming.frame().await {
                    None => {
                        if let Some(GiveBack { pool, key, sender }) = give_back.take() {
                            pool.put(&key, Sender::H1(sender));
                        }
                        return Ok(None);
                    }
                    Some(Err(error)) => return Err(failed(&error)),
                    Some(Ok(frame)) => {
                        if let Ok(data) = frame.into_data() {
                            return Ok(Some(data));
                        }
                    }
                }
            },
            Body::H3(stream) => match stream.recv_data().await {
                Ok(Some(mut data)) => Ok(Some(data.copy_to_bytes(data.remaining()))),
                Ok(None) => Ok(None),
                Err(error) => Err(failed(&error)),
            },
        }
    }
}

/// The shared client: TLS configurations, pool, caches.
pub struct Client {
    /// Where requests run: the fetch runtime, or (tests) the caller's.
    runtime: Option<tokio::runtime::Handle>,
    tcp_tls: Arc<rustls::ClientConfig>,
    quic: quinn::ClientConfig,
    /// QUIC endpoints, IPv4 and IPv6, opened on first use.
    endpoints: Mutex<[Option<quinn::Endpoint>; 2]>,
    pool: Arc<Pool>,
    alternatives: altsvc::Cache,
    pub(super) resolver: dns::Resolver,
    timeouts: Timeouts,
    /// Test-only: loopback addresses count as permitted, so local servers
    /// can stand in for public ones.
    #[cfg(test)]
    loopback: bool,
}

struct Global {
    /// Kept alive for the life of the process.
    _runtime: tokio::runtime::Runtime,
    client: Arc<Client>,
}

static GLOBAL: OnceLock<Result<Global, String>> = OnceLock::new();

/// The process-wide client, on its own runtime.
pub fn global() -> Result<Arc<Client>, ToolError> {
    GLOBAL
        .get_or_init(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("local-ai-fetch")
                .enable_all()
                .build()
                .map_err(|error| error.to_string())?;
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            let verifier = rustls_platform_verifier::Verifier::new(Arc::clone(&provider))
                .map_err(|error| error.to_string())?;
            let client = Client::new(
                Some(runtime.handle().clone()),
                provider,
                Arc::new(verifier),
                Timeouts::default(),
            )?;
            Ok(Global {
                _runtime: runtime,
                client: Arc::new(client),
            })
        })
        .as_ref()
        .map(|global| Arc::clone(&global.client))
        .map_err(|error| ToolError::Failed(format!("web fetch is unavailable: {error}")))
}

/// Aborts a spawned request when the caller stops waiting for it.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl Client {
    /// A client verifying certificates with `verifier`, running requests on
    /// `runtime` (or the caller's, when `None`).
    pub fn new(
        runtime: Option<tokio::runtime::Handle>,
        provider: Arc<rustls::crypto::CryptoProvider>,
        verifier: Arc<dyn rustls::client::danger::ServerCertVerifier>,
        timeouts: Timeouts,
    ) -> Result<Self, String> {
        let shared = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|error| error.to_string())?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        let mut tcp = shared.clone();
        tcp.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        tcp.resumption = rustls::client::Resumption::in_memory_sessions(SESSIONS);
        let mut quic = shared;
        quic.alpn_protocols = vec![b"h3".to_vec()];
        quic.resumption = rustls::client::Resumption::in_memory_sessions(SESSIONS);
        quic.enable_early_data = true;
        let quic = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(quic))
            .map_err(|error| error.to_string())?;
        let mut quic = quinn::ClientConfig::new(Arc::new(quic));
        let mut transport = quinn::TransportConfig::default();
        transport.max_idle_timeout(Some(IDLE.try_into().map_err(|_| "idle timeout")?));
        quic.transport_config(Arc::new(transport));
        Ok(Self {
            runtime,
            tcp_tls: Arc::new(tcp),
            quic,
            endpoints: Mutex::new([None, None]),
            pool: Arc::default(),
            alternatives: altsvc::Cache::default(),
            resolver: dns::Resolver::default(),
            timeouts,
            #[cfg(test)]
            loopback: false,
        })
    }

    /// Test-only: let loopback addresses through the address check.
    #[cfg(test)]
    #[must_use]
    pub const fn allow_loopback(mut self) -> Self {
        self.loopback = true;
        self
    }

    /// Test-only: forget pooled connections (alternatives are kept).
    #[cfg(test)]
    pub fn clear_pool(&self) {
        self.pool.origins().clear();
    }

    /// Test-only: the HTTP/3 port known for `key`.
    #[cfg(test)]
    pub fn h3_port(&self, key: &Key) -> Option<u16> {
        self.alternatives.h3_port(key)
    }

    /// Whether `ip` may be connected to.
    #[cfg_attr(
        not(test),
        expect(clippy::unused_self, reason = "uses the test switch in tests")
    )]
    fn permitted(&self, ip: IpAddr) -> bool {
        #[cfg(test)]
        if self.loopback && ip.is_loopback() {
            return true;
        }
        super::is_public(ip)
    }

    /// `GET` `target`, asking for `accept`, and return the response once its
    /// headers have arrived.
    pub async fn get(
        self: &Arc<Self>,
        target: &Target,
        accept: &'static str,
    ) -> Result<Response, ToolError> {
        let Some(runtime) = &self.runtime else {
            return Arc::clone(self).get_here(target.clone(), accept).await;
        };
        let mut task =
            AbortOnDrop(runtime.spawn(Arc::clone(self).get_here(target.clone(), accept)));
        (&mut task.0)
            .await
            .map_err(|error| ToolError::Failed(format!("fetch task failed: {error}")))?
    }

    async fn get_here(
        self: Arc<Self>,
        target: Target,
        accept: &'static str,
    ) -> Result<Response, ToolError> {
        let key = Key::of(&target);
        if let Some(sender) = self.pool.checkout(&key) {
            // A pooled connection the server closed meanwhile fails before
            // any response; `GET` is idempotent, so try a new one.
            if let Ok(response) = self.send(&key, &target, sender, accept).await {
                return Ok(response);
            }
        }
        let (sender, early) = self.connect(&key, &target, true).await?;
        match self.send(&key, &target, sender, accept).await {
            Err(_) if early => {
                // 0-RTT data the server rejected (or a resumed handshake
                // that failed): once more with a full handshake.
                let (sender, _) = self.connect(&key, &target, false).await?;
                self.send(&key, &target, sender, accept).await
            }
            result => result,
        }
    }

    /// Send `GET target` on `sender`.
    async fn send(
        &self,
        key: &Key,
        target: &Target,
        sender: Sender,
        accept: &'static str,
    ) -> Result<Response, ToolError> {
        let protocol = sender.protocol();
        let builder = if protocol == Protocol::Http1 {
            http::Request::get(target.path.as_str()).header(http::header::HOST, host_header(target))
        } else {
            let scheme = if target.https { "https" } else { "http" };
            http::Request::get(format!("{scheme}://{}{}", host_header(target), target.path))
        };
        let builder = builder
            .header(http::header::USER_AGENT, "local-ai-tools/1")
            .header(http::header::ACCEPT, accept)
            .header(http::header::ACCEPT_ENCODING, "zstd, br, gzip, deflate");
        let (status, headers, body) = match sender {
            Sender::H1(mut sender) => {
                let request = builder.body(Empty::new()).map_err(|error| failed(&error))?;
                sender.ready().await.map_err(|error| failed(&error))?;
                let response = sender
                    .send_request(request)
                    .await
                    .map_err(|error| failed(&error))?;
                let (parts, incoming) = response.into_parts();
                let reusable = !parts
                    .headers
                    .get(http::header::CONNECTION)
                    .and_then(|value| value.to_str().ok())
                    .is_some_and(|value| value.to_ascii_lowercase().contains("close"));
                let give_back = reusable.then(|| GiveBack {
                    pool: Arc::clone(&self.pool),
                    key: key.clone(),
                    sender,
                });
                (
                    parts.status,
                    parts.headers,
                    Body::Hyper(incoming, give_back),
                )
            }
            Sender::H2(mut sender) => {
                let request = builder.body(Empty::new()).map_err(|error| failed(&error))?;
                sender.ready().await.map_err(|error| failed(&error))?;
                let response = sender
                    .send_request(request)
                    .await
                    .map_err(|error| failed(&error))?;
                let (parts, incoming) = response.into_parts();
                (parts.status, parts.headers, Body::Hyper(incoming, None))
            }
            Sender::H3(mut sender, _) => {
                let request = builder.body(()).map_err(|error| failed(&error))?;
                let mut stream = sender
                    .send_request(request)
                    .await
                    .map_err(|error| failed(&error))?;
                stream.finish().await.map_err(|error| failed(&error))?;
                let response = stream
                    .recv_response()
                    .await
                    .map_err(|error| failed(&error))?;
                let (parts, ()) = response.into_parts();
                (parts.status, parts.headers, Body::H3(Box::new(stream)))
            }
        };
        if target.https {
            self.alternatives.learn(
                key,
                headers
                    .get_all(http::header::ALT_SVC)
                    .iter()
                    .filter_map(|value| value.to_str().ok()),
            );
        }
        Ok(Response {
            status,
            headers,
            #[cfg(test)]
            protocol,
            body,
        })
    }

    /// A new connection to `key`: QUIC raced against TCP when the origin is
    /// known to serve HTTP/3, else TCP. The flag says the connection is a
    /// 0-RTT one whose early data the server may still reject.
    async fn connect(
        self: &Arc<Self>,
        key: &Key,
        target: &Target,
        allow_early: bool,
    ) -> Result<(Sender, bool), ToolError> {
        // Tests never ask the real DNS service.
        let https_rr = (!cfg!(test)
            && target.https
            && target.host.parse::<IpAddr>().is_err()
            && self.alternatives.https_rr_due(key))
        .then(|| {
            let client = Arc::clone(self);
            let key = key.clone();
            tokio::task::spawn_blocking(move || {
                let found = dns::lookup_https(&key.host, HTTPS_RR_TIMEOUT);
                client.alternatives.looked_up(
                    &key,
                    found.map(|https| (https.port.unwrap_or(key.port), https.ttl)),
                );
            })
        });
        let addresses = self
            .resolver
            .resolve(&target.host, |ip| self.permitted(ip))
            .await?;
        if let Some(lookup) = https_rr {
            // Give the first fetch a chance to use HTTP/3; a slower answer
            // still lands in the cache for the next one.
            let _ = tokio::time::timeout(HTTPS_RR_GRACE, lookup).await;
        }
        let Some(port) = target
            .https
            .then(|| self.alternatives.h3_port(key))
            .flatten()
        else {
            return self
                .connect_tcp(key, target, &addresses)
                .await
                .map(|sender| (sender, false));
        };
        let mut quic = Box::pin(Arc::clone(self).connect_quic(
            key.clone(),
            Arc::clone(&addresses),
            port,
            allow_early,
        ));
        let tcp = async {
            tokio::time::sleep(self.timeouts.head_start).await;
            self.connect_tcp(key, target, &addresses).await
        };
        let mut tcp = std::pin::pin!(tcp);
        tokio::select! {
            quic_result = &mut quic => match quic_result {
                Ok(connected) => Ok(connected),
                Err(_) => self.connect_tcp(key, target, &addresses).await.map(|sender| (sender, false)),
            },
            tcp_result = &mut tcp => match tcp_result {
                Ok(sender) => {
                    // Let QUIC finish for later requests (it pools itself,
                    // or marks the origin broken), bounded by its timeout.
                    tokio::spawn(async move {
                        let _ = quic.await;
                    });
                    Ok((sender, false))
                }
                Err(error) => quic.await.map_err(|_| error),
            },
        }
    }

    /// A QUIC connection to one of `addresses` on `port`, with HTTP/3 on
    /// it, pooled. Failure marks the origin broken.
    async fn connect_quic(
        self: Arc<Self>,
        key: Key,
        addresses: Arc<[IpAddr]>,
        port: u16,
        allow_early: bool,
    ) -> Result<(Sender, bool), ToolError> {
        let endpoints: Vec<SocketAddr> = addresses
            .iter()
            .map(|ip| SocketAddr::new(*ip, port))
            .collect();
        let client = Arc::clone(&self);
        let host = key.host.clone();
        let connected = staggered(&endpoints, move |address| {
            let client = Arc::clone(&client);
            let host = host.clone();
            async move { client.quic_handshake(address, &host, allow_early).await }
        })
        .await;
        match connected {
            Ok((connection, early)) => {
                let h3 = h3::client::builder()
                    .max_field_section_size(u64::from(MAX_HEADER_BYTES))
                    .build::<_, _, Bytes>(h3_quinn::Connection::new(connection.clone()))
                    .await;
                let (mut driver, sender) = match h3 {
                    Ok(pair) => pair,
                    Err(error) => {
                        self.alternatives.mark_broken(&key);
                        return Err(failed(&error));
                    }
                };
                tokio::spawn(async move {
                    let _ = driver.wait_idle().await;
                });
                if !early {
                    self.alternatives.confirm(&key);
                }
                let sender = Sender::H3(sender, connection);
                if let Some(shared) = sender.share() {
                    self.pool.put(&key, shared);
                }
                Ok((sender, early))
            }
            Err(error) => {
                self.alternatives.mark_broken(&key);
                Err(ToolError::Failed(format!(
                    "fetch failed over QUIC: {error}"
                )))
            }
        }
    }

    /// One QUIC handshake (or 0-RTT start) with `address`.
    async fn quic_handshake(
        &self,
        address: SocketAddr,
        host: &str,
        allow_early: bool,
    ) -> Result<(quinn::Connection, bool), String> {
        if !self.permitted(address.ip()) {
            return Err(format!("{} is not a permitted address", address.ip()));
        }
        let endpoint = self.endpoint(address.is_ipv4())?;
        let connecting = endpoint
            .connect_with(self.quic.clone(), address, host)
            .map_err(|error| error.to_string())?;
        let connecting = if allow_early {
            match connecting.into_0rtt() {
                Ok((connection, accepted)) => {
                    // Close a 0-RTT connection whose handshake never
                    // completes, so its request fails (and is retried)
                    // instead of waiting for the idle timeout.
                    let watched = connection.clone();
                    let limit = self.timeouts.quic_handshake;
                    tokio::spawn(async move {
                        if tokio::time::timeout(limit, accepted).await.is_err() {
                            watched.close(0u32.into(), b"handshake timeout");
                        }
                    });
                    return Ok((connection, true));
                }
                Err(connecting) => connecting,
            }
        } else {
            connecting
        };
        tokio::time::timeout(self.timeouts.quic_handshake, connecting)
            .await
            .map_err(|_| "the QUIC handshake timed out".to_owned())?
            .map(|connection| (connection, false))
            .map_err(|error| error.to_string())
    }

    /// The QUIC endpoint for one address family, opened on first use.
    fn endpoint(&self, ipv4: bool) -> Result<quinn::Endpoint, String> {
        let mut endpoints = self
            .endpoints
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let slot = endpoints
            .get_mut(usize::from(!ipv4))
            .ok_or("no endpoint slot")?;
        if let Some(endpoint) = slot {
            return Ok(endpoint.clone());
        }
        let local: SocketAddr = if ipv4 {
            (std::net::Ipv4Addr::UNSPECIFIED, 0).into()
        } else {
            (std::net::Ipv6Addr::UNSPECIFIED, 0).into()
        };
        let endpoint = quinn::Endpoint::client(local).map_err(|error| error.to_string())?;
        *slot = Some(endpoint.clone());
        drop(endpoints);
        Ok(endpoint)
    }

    /// A TCP connection to one of `addresses`, with TLS and HTTP/2 or
    /// HTTP/1.1 on it as ALPN chooses.
    async fn connect_tcp(
        &self,
        key: &Key,
        target: &Target,
        addresses: &[IpAddr],
    ) -> Result<Sender, ToolError> {
        let endpoints: Vec<SocketAddr> = addresses
            .iter()
            .filter(|ip| self.permitted(**ip))
            .map(|ip| SocketAddr::new(*ip, target.port))
            .collect();
        let tcp = staggered(&endpoints, |address| async move {
            tokio::net::TcpStream::connect(address)
                .await
                .map_err(|error| error.to_string())
        })
        .await
        .map_err(|error| ToolError::Failed(format!("fetch failed: {error}")))?;
        let _ = tcp.set_nodelay(true);
        if !target.https {
            return handshake_h1(tcp).await;
        }
        let name = ServerName::try_from(target.host.clone()).map_err(|error| failed(&error))?;
        let tls = tokio_rustls::TlsConnector::from(Arc::clone(&self.tcp_tls))
            .connect(name, tcp)
            .await
            .map_err(|error| failed(&error))?;
        if tls.get_ref().1.alpn_protocol() != Some(b"h2") {
            return handshake_h1(tls).await;
        }
        let (sender, connection) = http2::Builder::new(TokioExecutor::new())
            .max_header_list_size(MAX_HEADER_BYTES)
            .handshake(TokioIo::new(tls))
            .await
            .map_err(|error| failed(&error))?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let sender = Sender::H2(sender);
        if let Some(shared) = sender.share() {
            self.pool.put(key, shared);
        }
        Ok(sender)
    }
}

async fn handshake_h1<T>(io: T) -> Result<Sender, ToolError>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (sender, connection) = http1::handshake(TokioIo::new(io))
        .await
        .map_err(|error| failed(&error))?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    Ok(Sender::H1(sender))
}

/// The first success of `attempt` over `addresses`, each started
/// [`STAGGER`] after the previous one or as soon as it fails; the losers are
/// abandoned.
async fn staggered<T, F, Fut>(addresses: &[SocketAddr], attempt: F) -> Result<T, String>
where
    T: Send + 'static,
    F: Fn(SocketAddr) -> Fut,
    Fut: Future<Output = Result<T, String>> + Send + 'static,
{
    let mut remaining = addresses.iter();
    let mut running = tokio::task::JoinSet::new();
    let mut last_error = "no permitted addresses".to_owned();
    loop {
        if running.is_empty() {
            match remaining.next() {
                Some(address) => {
                    running.spawn(attempt(*address));
                }
                None => return Err(last_error),
            }
        }
        let more = !remaining.as_slice().is_empty();
        tokio::select! {
            joined = running.join_next() => match joined {
                Some(Ok(Ok(value))) => return Ok(value),
                Some(Ok(Err(error))) => last_error = error,
                Some(Err(error)) => last_error = error.to_string(),
                None => {}
            },
            () = tokio::time::sleep(STAGGER), if more => {
                if let Some(address) = remaining.next() {
                    running.spawn(attempt(*address));
                }
            }
        }
    }
}

fn failed(error: &dyn std::fmt::Display) -> ToolError {
    ToolError::Failed(format!("fetch failed: {error}"))
}
