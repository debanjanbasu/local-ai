//! Audio encoders for synthesized speech (mono, 24 kHz).
//!
//! All are pure Rust under permissive licences: WAV and raw PCM here, FLAC with `flacenc`
//! (Apache-2.0), MP3 with `rusty_mp3` (Apache-2.0), Opus with `rusty-opus`
//! (BSD-3-Clause) in Ogg pages from the `ogg` crate (BSD-3-Clause). AAC has
//! no permissive encoder available and is refused.
//!
//! [`StreamEncoder`] takes samples chunk by chunk and returns the bytes that
//! are ready, so responses can stream: PCM, WAV (with an open-ended length
//! in its header), MP3 (whole frames) and Opus (whole Ogg pages) stream
//! progressively; FLAC needs the whole signal for its header and arrives at
//! [`StreamEncoder::finish`].

use std::io::Cursor;

use crate::Error;

/// MP3 bitrate (MPEG-2 Layer III at 24 kHz).
const MP3_KBPS: u32 = 64;
/// Opus bitrate and frame (20 ms at 24 kHz).
const OPUS_BPS: i32 = 32_000;
const OPUS_FRAME: usize = 480;
/// Encoder lookahead at 48 kHz written as the Ogg Opus pre-skip.
const OPUS_PRE_SKIP: u16 = 312;
const OGG_SERIAL: u32 = 0x4b4f_4b4f;

/// The pinned API's `response_format` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl Format {
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "mp3" => Self::Mp3,
            "opus" => Self::Opus,
            "aac" => Self::Aac,
            "flac" => Self::Flac,
            "wav" => Self::Wav,
            "pcm" => Self::Pcm,
            _ => return None,
        })
    }

    /// Whether this build can encode the format (AAC: no permissive encoder).
    #[must_use]
    pub const fn supported(self) -> bool {
        !matches!(self, Self::Aac)
    }

    #[must_use]
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/opus",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }
}

/// Clamp F32 samples to 16-bit PCM.
#[must_use]
pub fn pcm16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect()
}

/// Raw little-endian 16-bit PCM bytes (`response_format: pcm`).
#[must_use]
pub fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
    pcm16(samples)
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect()
}

/// A 44-byte RIFF/WAVE header for mono 16-bit PCM. `data_bytes` of
/// `u32::MAX` marks an unknown (streamed) length, as common encoders do.
#[must_use]
pub fn wav_header(sample_rate: u32, data_bytes: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&data_bytes.saturating_add(36).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16_u32.to_le_bytes());
    h.extend_from_slice(&1_u16.to_le_bytes());
    h.extend_from_slice(&1_u16.to_le_bytes());
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    h.extend_from_slice(&2_u16.to_le_bytes());
    h.extend_from_slice(&16_u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_bytes.to_le_bytes());
    h
}

/// A complete WAV file.
#[must_use]
pub fn wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data = pcm_bytes(samples);
    let mut out = wav_header(sample_rate, u32::try_from(data.len()).unwrap_or(u32::MAX));
    out.extend_from_slice(&data);
    out
}

fn failure(what: &str, error: impl std::fmt::Debug) -> Error {
    Error::Generation(format!("{what} encoding failed: {error:?}"))
}

enum State {
    Pcm,
    Wav { header_sent: bool },
    Flac { samples: Vec<i32> },
    Mp3(Box<rusty_mp3::Mp3Encoder>),
    Opus(Box<OpusState>),
}

struct OpusState {
    encoder: rusty_opus::OpusEncoder,
    writer: ogg::PacketWriter<'static, Cursor<Vec<u8>>>,
    pending: Vec<f32>,
    granule: u64,
    taken: usize,
}

/// Incremental encoder for one response.
pub struct StreamEncoder {
    format: Format,
    sample_rate: u32,
    state: State,
}

impl StreamEncoder {
    /// # Errors
    ///
    /// An unsupported format or an encoder that cannot be set up.
    pub fn new(format: Format, sample_rate: u32) -> crate::Result<Self> {
        let state = match format {
            Format::Pcm => State::Pcm,
            Format::Wav => State::Wav { header_sent: false },
            Format::Flac => State::Flac {
                samples: Vec::new(),
            },
            Format::Mp3 => State::Mp3(Box::new(rusty_mp3::Mp3Encoder::new(
                rusty_mp3::Mp3EncoderConfig {
                    bitrate_kbps: MP3_KBPS,
                    vbr_quality: None,
                },
            ))),
            Format::Opus => {
                let mut encoder = rusty_opus::OpusEncoder::new(
                    i32::try_from(sample_rate).map_err(|e| failure("Opus", e))?,
                    1,
                    rusty_opus::Application::Voip,
                )
                .map_err(|e| failure("Opus", e))?;
                encoder.bitrate_bps = OPUS_BPS;
                let mut writer = ogg::PacketWriter::new(Cursor::new(Vec::new()));
                let mut head = Vec::with_capacity(19);
                head.extend_from_slice(b"OpusHead");
                head.push(1);
                head.push(1);
                head.extend_from_slice(&OPUS_PRE_SKIP.to_le_bytes());
                head.extend_from_slice(&sample_rate.to_le_bytes());
                head.extend_from_slice(&0_i16.to_le_bytes());
                head.push(0);
                writer
                    .write_packet(head, OGG_SERIAL, ogg::PacketWriteEndInfo::EndPage, 0)
                    .map_err(|e| failure("Opus", e))?;
                let vendor = b"local-ai rusty-opus";
                let mut tags = Vec::new();
                tags.extend_from_slice(b"OpusTags");
                tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
                tags.extend_from_slice(vendor);
                tags.extend_from_slice(&0_u32.to_le_bytes());
                writer
                    .write_packet(tags, OGG_SERIAL, ogg::PacketWriteEndInfo::EndPage, 0)
                    .map_err(|e| failure("Opus", e))?;
                State::Opus(Box::new(OpusState {
                    encoder,
                    writer,
                    pending: Vec::new(),
                    granule: u64::from(OPUS_PRE_SKIP),
                    taken: 0,
                }))
            }
            Format::Aac => {
                return Err(Error::InvalidArgument(
                    "response_format aac is not supported: no permissively licensed AAC encoder \
                     is available; use mp3, opus, flac, wav or pcm"
                        .into(),
                ));
            }
        };
        Ok(Self {
            format,
            sample_rate,
            state,
        })
    }

    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    /// Encode more samples; returns the bytes ready to send.
    ///
    /// # Errors
    ///
    /// An encoder failure.
    pub fn push(&mut self, samples: &[f32]) -> crate::Result<Vec<u8>> {
        match &mut self.state {
            State::Pcm => Ok(pcm_bytes(samples)),
            State::Wav { header_sent } => {
                let mut out = Vec::new();
                if !*header_sent {
                    out = wav_header(self.sample_rate, u32::MAX);
                    *header_sent = true;
                }
                out.extend(pcm_bytes(samples));
                Ok(out)
            }
            State::Flac { samples: all } => {
                all.extend(pcm16(samples).into_iter().map(i32::from));
                Ok(Vec::new())
            }
            State::Mp3(encoder) => {
                encoder
                    .push_pcm_f32(samples, 1, self.sample_rate)
                    .map_err(|e| failure("MP3", e))?;
                Ok(drain_mp3(encoder))
            }
            State::Opus(state) => {
                state.pending.extend_from_slice(samples);
                state.encode_frames(false)?;
                Ok(state.take())
            }
        }
    }

    /// Flush the encoder; returns the remaining bytes.
    ///
    /// # Errors
    ///
    /// An encoder failure.
    pub fn finish(mut self) -> crate::Result<Vec<u8>> {
        match &mut self.state {
            State::Pcm => Ok(Vec::new()),
            State::Wav { header_sent } => Ok(if *header_sent {
                Vec::new()
            } else {
                wav_header(self.sample_rate, 0)
            }),
            State::Flac { samples } => {
                use flacenc::component::BitRepr as _;
                use flacenc::error::Verify as _;
                let config = flacenc::config::Encoder::default()
                    .into_verified()
                    .map_err(|e| failure("FLAC", e))?;
                let source = flacenc::source::MemSource::from_samples(
                    samples,
                    1,
                    16,
                    self.sample_rate as usize,
                );
                let stream =
                    flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
                        .map_err(|e| failure("FLAC", e))?;
                let mut sink = flacenc::bitsink::ByteSink::new();
                stream.write(&mut sink).map_err(|e| failure("FLAC", e))?;
                Ok(sink.as_slice().to_vec())
            }
            State::Mp3(encoder) => {
                encoder.finish();
                Ok(drain_mp3(encoder))
            }
            State::Opus(state) => {
                state.encode_frames(true)?;
                Ok(state.take())
            }
        }
    }
}

fn drain_mp3(encoder: &mut rusty_mp3::Mp3Encoder) -> Vec<u8> {
    let mut out = Vec::new();
    while let Ok(packet) = encoder.next_packet() {
        // A Xing/Info header produced at the end cannot be prepended to a
        // stream already sent; players read the frames without it.
        if packet.windows(4).any(|w| w == b"Xing" || w == b"Info")
            && packet.len() < 1500
            && is_info_frame(&packet)
        {
            continue;
        }
        out.extend_from_slice(&packet);
    }
    out
}

/// Whether an MP3 frame is a Xing/Info metadata frame (tag right after the
/// side information, MPEG-2 mono: 4 + 9 bytes).
fn is_info_frame(packet: &[u8]) -> bool {
    [13_usize, 21, 36].iter().any(|&at| {
        packet
            .get(at..at + 4)
            .is_some_and(|t| t == b"Xing" || t == b"Info")
    })
}

impl OpusState {
    fn encode_frames(&mut self, last: bool) -> crate::Result<()> {
        let mut packet = [0_u8; 1500];
        loop {
            let available = self.pending.len();
            if available < OPUS_FRAME && !(last && available > 0) {
                break;
            }
            let mut frame: Vec<f32> = self.pending.drain(..available.min(OPUS_FRAME)).collect();
            let real = frame.len();
            frame.resize(OPUS_FRAME, 0.0);
            let len = self
                .encoder
                .encode(&frame, OPUS_FRAME, &mut packet)
                .map_err(|e| failure("Opus", e))?;
            // Granule positions count 48 kHz samples, pre-skip included; the
            // final page's granule trims the zero padding of the last frame.
            self.granule += 2 * real as u64;
            let end = if last && self.pending.is_empty() {
                ogg::PacketWriteEndInfo::EndStream
            } else {
                ogg::PacketWriteEndInfo::EndPage
            };
            self.writer
                .write_packet(packet[..len].to_vec(), OGG_SERIAL, end, self.granule)
                .map_err(|e| failure("Opus", e))?;
        }
        Ok(())
    }

    fn take(&mut self) -> Vec<u8> {
        let all = self.writer.inner().get_ref();
        let out = all[self.taken..].to_vec();
        self.taken = all.len();
        out
    }
}

/// Encode a whole signal.
///
/// # Errors
///
/// An unsupported format or an encoder failure.
pub fn encode(samples: &[f32], format: Format, sample_rate: u32) -> crate::Result<Vec<u8>> {
    let mut encoder = StreamEncoder::new(format, sample_rate)?;
    let mut out = encoder.push(samples)?;
    out.extend(encoder.finish()?);
    if format == Format::Wav {
        // A complete file carries its real length.
        let data = u32::try_from(out.len() - 44).unwrap_or(u32::MAX);
        out[..44].copy_from_slice(&wav_header(sample_rate, data));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn tone() -> Vec<f32> {
        (0..24_000)
            .map(|n| 0.3 * (std::f32::consts::TAU * 440.0 * n as f32 / 24_000.0).sin())
            .collect()
    }

    #[test]
    fn wav_and_pcm_carry_every_sample() {
        let x = tone();
        let w = encode(&x, Format::Wav, 24_000).expect("wav");
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(w.len(), 44 + 2 * x.len());
        assert_eq!(
            u32::from_le_bytes([w[40], w[41], w[42], w[43]]) as usize,
            2 * x.len()
        );
        assert_eq!(
            encode(&x, Format::Pcm, 24_000).expect("pcm").len(),
            2 * x.len()
        );
    }

    #[test]
    fn compressed_formats_have_their_magic() {
        let x = tone();
        let flac = encode(&x, Format::Flac, 24_000).expect("flac");
        assert_eq!(&flac[..4], b"fLaC");
        let mp3 = encode(&x, Format::Mp3, 24_000).expect("mp3");
        assert_eq!(mp3[0], 0xFF);
        assert_eq!(mp3[1] & 0xE0, 0xE0);
        let opus = encode(&x, Format::Opus, 24_000).expect("opus");
        assert_eq!(&opus[..4], b"OggS");
        assert!(opus.windows(8).any(|w| w == b"OpusHead"));
        assert!(Format::parse("aac").is_some_and(|f| !f.supported()));
        assert!(StreamEncoder::new(Format::Aac, 24_000).is_err());
    }

    #[test]
    fn streaming_equals_whole_for_raw_formats() {
        let x = tone();
        let mut s = StreamEncoder::new(Format::Pcm, 24_000).expect("pcm");
        let mut out = s.push(&x[..1000]).expect("a");
        out.extend(s.push(&x[1000..]).expect("b"));
        out.extend(s.finish().expect("c"));
        assert_eq!(out, pcm_bytes(&x));
    }
}
