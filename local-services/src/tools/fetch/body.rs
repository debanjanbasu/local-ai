//! Reading a response body through its content coding (`zstd`, `br`,
//! `gzip`, `deflate` or none), streaming, within two limits: the decoded
//! bytes kept and the encoded bytes read. A decompression bomb stops at the
//! decoded limit (the output buffer refuses more and decoding ends there);
//! a stream that decodes to little or nothing stops at the encoded limit.
//! The zstd window is capped at 8 MiB (RFC 9659), Brotli's is at most
//! 16 MiB by its format, and deflate's at 32 KiB.

use std::io::Write as _;

use super::ToolError;
use super::client::Response;

/// Encoded bytes read beyond the decoded limit before stopping: framing and
/// incompressible data can make an encoded body slightly larger than the
/// bytes it decodes to.
const ENCODED_SLACK: usize = 64 * 1024;
/// The zstd window log accepted: 8 MiB, the HTTP limit (RFC 9659).
const ZSTD_WINDOW_LOG_MAX: u32 = 23;

/// A response's content coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coding {
    Identity,
    Gzip,
    Deflate,
    Brotli,
    Zstd,
}

impl Coding {
    /// The coding a `Content-Encoding` value names; layered or unknown
    /// codings are refused.
    pub fn parse(value: Option<&str>) -> Result<Self, ToolError> {
        let value = value.unwrap_or_default().trim().to_ascii_lowercase();
        Ok(match value.as_str() {
            "" | "identity" => Self::Identity,
            "gzip" | "x-gzip" => Self::Gzip,
            "deflate" => Self::Deflate,
            "br" => Self::Brotli,
            "zstd" => Self::Zstd,
            other => {
                return Err(ToolError::Failed(format!(
                    "the response uses the unsupported content coding {other:?}"
                )));
            }
        })
    }
}

/// A body read up to a limit.
#[derive(Debug)]
pub struct Read {
    /// The decoded bytes kept.
    pub bytes: Vec<u8>,
    /// Whether a limit stopped the read before the body ended.
    pub cut: bool,
}

/// Read `response`'s body, decoded, keeping at most `limit` bytes and
/// reading at most `limit` plus a small slack of encoded bytes.
pub async fn read(response: &mut Response, limit: usize) -> Result<Read, ToolError> {
    let coding = Coding::parse(
        response
            .headers
            .get(http::header::CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok()),
    )?;
    let encoded_limit = limit.saturating_add(ENCODED_SLACK);
    let mut decoder = Decoder::new(coding, limit)?;
    let mut encoded = 0;
    let mut cut = false;
    while let Some(chunk) = response.chunk().await? {
        let room = encoded_limit - encoded;
        let take = chunk.get(..room.min(chunk.len())).unwrap_or_default();
        encoded += take.len();
        if decoder.push(take)? || take.len() < chunk.len() {
            cut = true;
            break;
        }
    }
    let bytes = decoder.finish(cut)?;
    Ok(Read { bytes, cut })
}

/// Output buffer that keeps at most `limit` bytes and then refuses more,
/// which ends the decoder's write with an error that [`Decoder`] reads as
/// "full".
#[derive(Debug)]
struct Sink {
    bytes: Vec<u8>,
    limit: usize,
    full: bool,
}

impl std::io::Write for Sink {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let room = self.limit - self.bytes.len();
        if room == 0 && !data.is_empty() {
            self.full = true;
            return Err(std::io::Error::other("the decoded limit was reached"));
        }
        let take = data.get(..room.min(data.len())).unwrap_or_default();
        self.bytes.extend_from_slice(take);
        Ok(take.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

enum Decoder {
    Identity(Sink),
    Gzip(flate2::write::MultiGzDecoder<Sink>),
    /// `deflate` before its first two bytes say zlib or raw.
    Deflate(Sink, Vec<u8>),
    Zlib(flate2::write::ZlibDecoder<Sink>),
    RawDeflate(flate2::write::DeflateDecoder<Sink>),
    Brotli(Box<brotli_decompressor::DecompressorWriter<Sink>>),
    Zstd(zstd::stream::zio::Writer<Sink, zstd::stream::raw::Decoder<'static>>),
}

impl Decoder {
    fn new(coding: Coding, limit: usize) -> Result<Self, ToolError> {
        let sink = Sink {
            bytes: Vec::new(),
            limit,
            full: false,
        };
        Ok(match coding {
            Coding::Identity => Self::Identity(sink),
            Coding::Gzip => Self::Gzip(flate2::write::MultiGzDecoder::new(sink)),
            Coding::Deflate => Self::Deflate(sink, Vec::new()),
            Coding::Brotli => Self::Brotli(Box::new(brotli_decompressor::DecompressorWriter::new(
                sink, 4096,
            ))),
            Coding::Zstd => {
                let mut decoder = zstd::stream::raw::Decoder::new().map_err(corrupt)?;
                decoder
                    .set_parameter(zstd::stream::raw::DParameter::WindowLogMax(
                        ZSTD_WINDOW_LOG_MAX,
                    ))
                    .map_err(corrupt)?;
                Self::Zstd(zstd::stream::zio::Writer::new(sink, decoder))
            }
        })
    }

    fn sink(&self) -> &Sink {
        match self {
            Self::Identity(sink) | Self::Deflate(sink, _) => sink,
            Self::Gzip(decoder) => decoder.get_ref(),
            Self::Zlib(decoder) => decoder.get_ref(),
            Self::RawDeflate(decoder) => decoder.get_ref(),
            Self::Brotli(decoder) => decoder.get_ref(),
            Self::Zstd(decoder) => decoder.writer(),
        }
    }

    fn sink_mut(&mut self) -> &mut Sink {
        match self {
            Self::Identity(sink) | Self::Deflate(sink, _) => sink,
            Self::Gzip(decoder) => decoder.get_mut(),
            Self::Zlib(decoder) => decoder.get_mut(),
            Self::RawDeflate(decoder) => decoder.get_mut(),
            Self::Brotli(decoder) => decoder.get_mut(),
            Self::Zstd(decoder) => decoder.writer_mut(),
        }
    }

    /// Decode `data`; `true` once the decoded limit is reached.
    fn push(&mut self, data: &[u8]) -> Result<bool, ToolError> {
        if let Self::Deflate(sink, pending) = self {
            pending.extend_from_slice(data);
            let Some(&[cmf, flg]) = pending.first_chunk::<2>() else {
                return Ok(false);
            };
            // RFC 1950: compression method 8 and a header checksum; anything
            // else is the raw deflate some servers send for `deflate`.
            let zlib = cmf & 0x0f == 8 && (u16::from(cmf) << 8 | u16::from(flg)) % 31 == 0;
            let sink = std::mem::replace(
                sink,
                Sink {
                    bytes: Vec::new(),
                    limit: 0,
                    full: false,
                },
            );
            let pending = std::mem::take(pending);
            *self = if zlib {
                Self::Zlib(flate2::write::ZlibDecoder::new(sink))
            } else {
                Self::RawDeflate(flate2::write::DeflateDecoder::new(sink))
            };
            return self.push(&pending);
        }
        let written = match self {
            Self::Identity(sink) => sink.write_all(data),
            Self::Gzip(decoder) => decoder.write_all(data),
            Self::Zlib(decoder) => decoder.write_all(data),
            Self::RawDeflate(decoder) => decoder.write_all(data),
            Self::Brotli(decoder) => decoder.write_all(data),
            Self::Zstd(decoder) => decoder.write_all(data),
            Self::Deflate(..) => Ok(()),
        };
        match written {
            Ok(()) => Ok(self.sink().full),
            Err(_) if self.sink().full => Ok(true),
            Err(error) => Err(corrupt(error)),
        }
    }

    /// The decoded bytes. After a whole body (`cut` false) the stream must
    /// end cleanly; after a cut one, what was decoded so far is kept.
    fn finish(mut self, cut: bool) -> Result<Vec<u8>, ToolError> {
        let ended = match &mut self {
            Self::Identity(_) => Ok(()),
            // Fewer than two bytes of `deflate`: an empty body is fine.
            Self::Deflate(_, pending) => {
                if pending.is_empty() {
                    Ok(())
                } else {
                    Err(std::io::Error::other("a truncated deflate stream"))
                }
            }
            Self::Gzip(decoder) => decoder.try_finish(),
            Self::Zlib(decoder) => decoder.try_finish(),
            Self::RawDeflate(decoder) => decoder.try_finish(),
            Self::Brotli(decoder) => decoder.close(),
            Self::Zstd(decoder) => decoder.finish(),
        };
        if let Err(error) = ended
            && !cut
            && !self.sink().full
        {
            return Err(corrupt(error));
        }
        Ok(std::mem::take(&mut self.sink_mut().bytes))
    }
}

fn corrupt(error: impl std::fmt::Display) -> ToolError {
    ToolError::Failed(format!("cannot decode the response body: {error}"))
}

#[cfg(test)]
pub mod tests {
    use super::{Coding, Decoder};

    /// Encode `data` with `coding`.
    pub fn encode(coding: Coding, data: &[u8]) -> Vec<u8> {
        use std::io::Write as _;
        match coding {
            Coding::Identity => data.to_vec(),
            Coding::Gzip => {
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                encoder.write_all(data).expect("gzip");
                encoder.finish().expect("gzip")
            }
            Coding::Deflate => {
                let mut encoder =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
                encoder.write_all(data).expect("zlib");
                encoder.finish().expect("zlib")
            }
            Coding::Brotli => {
                let mut out = Vec::new();
                brotli::BrotliCompress(
                    &mut std::io::Cursor::new(data),
                    &mut out,
                    &brotli::enc::BrotliEncoderParams {
                        quality: 1,
                        ..Default::default()
                    },
                )
                .expect("brotli");
                out
            }
            Coding::Zstd => zstd::encode_all(data, 1).expect("zstd"),
        }
    }

    fn decode(coding: Coding, encoded: &[u8], limit: usize, chunk: usize) -> (Vec<u8>, bool) {
        let mut decoder = Decoder::new(coding, limit).expect("decoder");
        let mut cut = false;
        for piece in encoded.chunks(chunk.max(1)) {
            if decoder.push(piece).expect("decodes") {
                cut = true;
                break;
            }
        }
        (decoder.finish(cut).expect("finish"), cut)
    }

    #[test]
    fn every_coding_round_trips_in_any_chunking() {
        let text: Vec<u8> = (0..20_000u32)
            .flat_map(|n| format!("line {n} of the sample body\n").into_bytes())
            .collect();
        for coding in [
            Coding::Identity,
            Coding::Gzip,
            Coding::Deflate,
            Coding::Brotli,
            Coding::Zstd,
        ] {
            let encoded = encode(coding, &text);
            for chunk in [1, 7, 4096, usize::MAX] {
                assert_eq!(
                    decode(coding, &encoded, usize::MAX, chunk),
                    (text.clone(), false),
                    "{coding:?} in chunks of {chunk}"
                );
            }
            let (cut, was_cut) = decode(coding, &encoded, 1000, 4096);
            assert!(was_cut, "{coding:?}");
            assert_eq!(cut, text.get(..1000).expect("prefix"), "{coding:?}");
        }
        // Raw deflate, as some servers send for `deflate`.
        let mut raw = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
        std::io::Write::write_all(&mut raw, &text).expect("deflate");
        let raw = raw.finish().expect("deflate");
        assert_eq!(decode(Coding::Deflate, &raw, usize::MAX, 1), (text, false));
    }

    #[test]
    fn corrupt_and_unknown_codings_fail() {
        let mut decoder = Decoder::new(Coding::Gzip, 1 << 20).expect("decoder");
        assert!(decoder.push(b"definitely not gzip").is_err());
        let truncated = encode(Coding::Zstd, &vec![7; 100_000]);
        let mut decoder = Decoder::new(Coding::Zstd, 1 << 20).expect("decoder");
        decoder
            .push(truncated.get(..truncated.len() / 2).expect("half"))
            .expect("decodes so far");
        assert!(decoder.finish(false).is_err());
        assert!(Coding::parse(Some("gzip, br")).is_err());
        assert!(Coding::parse(Some("compress")).is_err());
        assert_eq!(Coding::parse(Some(" BR ")).expect("br"), Coding::Brotli);
        assert_eq!(Coding::parse(None).expect("none"), Coding::Identity);
    }

    #[test]
    fn bombs_stop_at_the_decoded_limit() {
        let zeros = vec![0u8; 16 << 20];
        for coding in [Coding::Gzip, Coding::Brotli, Coding::Zstd, Coding::Deflate] {
            let bomb = encode(coding, &zeros);
            assert!(bomb.len() < 1 << 20, "{coding:?}: {}", bomb.len());
            let started = std::time::Instant::now();
            let (kept, cut) = decode(coding, &bomb, 64 * 1024, 16 * 1024);
            assert!(cut && kept.len() == 64 * 1024, "{coding:?}");
            assert!(started.elapsed().as_secs() < 2, "{coding:?}");
        }
        // A zstd window over 8 MiB is refused rather than allocated.
        let mut wide = zstd::stream::write::Encoder::new(Vec::new(), 1).expect("encoder");
        wide.window_log(27).expect("window");
        wide.long_distance_matching(true).expect("ldm");
        std::io::Write::write_all(&mut wide, &zeros).expect("encode");
        let wide = wide.finish().expect("finish");
        let mut decoder = Decoder::new(Coding::Zstd, usize::MAX).expect("decoder");
        assert!(decoder.push(&wide).is_err());
    }
}
