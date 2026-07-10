//! **Remote voice backends** — offload TTS synthesis and STT transcription to a
//! hosted provider (ElevenLabs, Cartesia, Inworld, Sarvam, or any OpenAI-
//! compatible `/v1/audio/*` endpoint) instead of the in-kernel ONNX models.
//!
//! Same posture as the remote *chat* backend ([`super::remote`]): the API key +
//! provider are **human-configured** (`/voice remote …`, never an agent tool),
//! persisted at `/configs/core/voice.json`. Only the model call leaves the box;
//! the audio pipeline (chunked playback, the mic front-end, Ctrl+C) is
//! unchanged — remote TTS returns PCM into the very same `speech_pump` queue,
//! so streaming-per-clause playback works identically to the local path.
//!
//! Wire code is deterministic native (below the boundary): each provider is a
//! request builder + a response decoder. Audio comes back as WAV/MP3 bytes
//! (decoded via [`crate::audio::decode`]) or base64 in a JSON field
//! (Inworld/Sarvam) — resampled to the device rate. STT uploads the utterance
//! as a `multipart/form-data` WAV and reads the transcript field.
//!
//! **TLS caveat:** every provider is HTTPS, and the in-kernel TLS client
//! (`net/tls.rs`, embedded-tls, no cert verification) does not interop with all
//! servers yet (RSA cert chains fail). A provider that won't connect reports a
//! TLS error here, not a wrong result.

use crate::json::Json;
use crate::net::http;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;

const VOICE_CFG_PATH: &str = "/configs/core/voice.json";
/// Synthesis/transcription can take a few seconds on a long clause; generous.
const HTTP_TIMEOUT_MS: u64 = 60_000;

/// A supported hosted voice provider. `parse`/`name` keep the on-disk config
/// and the `/voice remote` command in lockstep.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    ElevenLabs,
    Cartesia,
    Inworld,
    Sarvam,
    /// Any OpenAI-compatible `/v1/audio/{speech,transcriptions}` server
    /// (OpenAI, Groq, local gateways). Base URL supplied via the model field
    /// as `url@model` when it isn't `api.openai.com`.
    OpenAI,
}

impl Provider {
    pub fn parse(s: &str) -> Option<Provider> {
        Some(match s.to_ascii_lowercase().as_str() {
            "elevenlabs" | "11labs" | "eleven" => Provider::ElevenLabs,
            "cartesia" => Provider::Cartesia,
            "inworld" => Provider::Inworld,
            "sarvam" => Provider::Sarvam,
            "openai" | "groq" | "compatible" => Provider::OpenAI,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Provider::ElevenLabs => "elevenlabs",
            Provider::Cartesia => "cartesia",
            Provider::Inworld => "inworld",
            Provider::Sarvam => "sarvam",
            Provider::OpenAI => "openai",
        }
    }
}

/// One direction's remote config (TTS or STT share the shape).
#[derive(Clone)]
pub struct Endpoint {
    pub provider: Provider,
    pub key: String,
    /// Voice id / speaker (TTS). Provider default when empty.
    pub voice: String,
    /// Model id, optionally `base_url@model` for the OpenAI-compatible case.
    pub model: String,
}

impl Endpoint {
    /// `(base_url, model)` — splits the `url@model` form for OpenAI-compatible.
    fn base_and_model(&self, default_base: &str, default_model: &str) -> (String, String) {
        if let Some((b, m)) = self.model.split_once('@') {
            (b.trim_end_matches('/').to_string(), m.to_string())
        } else {
            (default_base.to_string(), if self.model.is_empty() { default_model.to_string() } else { self.model.clone() })
        }
    }
}

/// The persisted remote-voice config: independent TTS and STT endpoints.
#[derive(Clone, Default)]
pub struct VoiceConfig {
    pub tts: Option<Endpoint>,
    pub stt: Option<Endpoint>,
}

fn parse_endpoint(j: &Json) -> Option<Endpoint> {
    let provider = Provider::parse(j.get("provider")?.as_str()?)?;
    let key = j.get("key").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if key.is_empty() {
        return None;
    }
    Some(Endpoint {
        provider,
        key,
        voice: j.get("voice").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        model: j.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
    })
}

/// Parse `voice.json` → `VoiceConfig`. Pure — unit-tested.
pub fn parse_config_json(bytes: &[u8]) -> VoiceConfig {
    let Some(j) = core::str::from_utf8(bytes).ok().and_then(Json::parse) else {
        return VoiceConfig::default();
    };
    VoiceConfig {
        tts: j.get("tts").and_then(parse_endpoint),
        stt: j.get("stt").and_then(parse_endpoint),
    }
}

fn endpoint_json(e: &Endpoint) -> Json {
    Json::Obj(vec![
        ("provider".to_string(), Json::Str(e.provider.name().to_string())),
        ("key".to_string(), Json::Str(e.key.clone())),
        ("voice".to_string(), Json::Str(e.voice.clone())),
        ("model".to_string(), Json::Str(e.model.clone())),
    ])
}

pub fn load() -> VoiceConfig {
    crate::synapse::fs::read(VOICE_CFG_PATH).map(|b| parse_config_json(&b)).unwrap_or_default()
}

pub fn save(cfg: &VoiceConfig) {
    let mut obj = Vec::new();
    if let Some(t) = &cfg.tts {
        obj.push(("tts".to_string(), endpoint_json(t)));
    }
    if let Some(s) = &cfg.stt {
        obj.push(("stt".to_string(), endpoint_json(s)));
    }
    crate::synapse::fs::write(VOICE_CFG_PATH, Json::Obj(obj).to_pretty().as_bytes());
}

// --- TTS ---------------------------------------------------------------------

/// A built TTS request: URL, headers, JSON body, and how to read the response.
pub struct TtsRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
    /// How the audio comes back.
    pub decode: RespKind,
}

/// How to extract audio bytes from a TTS response.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RespKind {
    /// Body is raw audio (WAV/MP3) — decode directly.
    AudioBytes,
    /// Body is JSON; the audio is base64 under this dotted field
    /// (Inworld `audioContent`, Sarvam `audios.0`).
    JsonBase64(&'static str),
}

/// Build the provider-specific TTS request for `text`. Pure — unit-tested, so
/// the exact wire shape per provider is checked without a network.
pub fn build_tts(e: &Endpoint, text: &str) -> TtsRequest {
    let t = Json::Str(text.to_string()).to_pretty(); // JSON-escaped, quoted
    match e.provider {
        Provider::ElevenLabs => {
            let voice = if e.voice.is_empty() { "JBFqnCBsd6RMkjVDRZzb" } else { &e.voice };
            let model = if e.model.is_empty() { "eleven_multilingual_v2" } else { &e.model };
            TtsRequest {
                // pcm_24000 → a raw-PCM WAV we wrap below; simplest to decode.
                url: format!("https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000"),
                headers: vec![("xi-api-key".to_string(), e.key.clone()), ("Content-Type".to_string(), "application/json".to_string())],
                body: format!("{{\"text\":{t},\"model_id\":\"{model}\"}}"),
                decode: RespKind::AudioBytes,
            }
        }
        Provider::Cartesia => {
            let voice = if e.voice.is_empty() { "a0e99841-438c-4a64-b679-ae501e7d6091" } else { &e.voice };
            let model = if e.model.is_empty() { "sonic-2" } else { &e.model };
            TtsRequest {
                url: "https://api.cartesia.ai/tts/bytes".to_string(),
                headers: vec![
                    ("X-API-Key".to_string(), e.key.clone()),
                    ("Cartesia-Version".to_string(), "2024-06-10".to_string()),
                    ("Content-Type".to_string(), "application/json".to_string()),
                ],
                body: format!(
                    "{{\"model_id\":\"{model}\",\"transcript\":{t},\"voice\":{{\"mode\":\"id\",\"id\":\"{voice}\"}},\"output_format\":{{\"container\":\"wav\",\"encoding\":\"pcm_s16le\",\"sample_rate\":24000}}}}"
                ),
                decode: RespKind::AudioBytes,
            }
        }
        Provider::Inworld => {
            let voice = if e.voice.is_empty() { "Ashley" } else { &e.voice };
            let model = if e.model.is_empty() { "inworld-tts-1" } else { &e.model };
            TtsRequest {
                url: "https://api.inworld.ai/tts/v1/voice".to_string(),
                headers: vec![
                    ("Authorization".to_string(), format!("Basic {}", e.key)),
                    ("Content-Type".to_string(), "application/json".to_string()),
                ],
                body: format!("{{\"text\":{t},\"voiceId\":\"{voice}\",\"modelId\":\"{model}\"}}"),
                decode: RespKind::JsonBase64("audioContent"),
            }
        }
        Provider::Sarvam => {
            let speaker = if e.voice.is_empty() { "anushka" } else { &e.voice };
            let model = if e.model.is_empty() { "bulbul:v2" } else { &e.model };
            TtsRequest {
                url: "https://api.sarvam.ai/text-to-speech".to_string(),
                headers: vec![
                    ("api-subscription-key".to_string(), e.key.clone()),
                    ("Content-Type".to_string(), "application/json".to_string()),
                ],
                body: format!(
                    "{{\"text\":{t},\"target_language_code\":\"en-IN\",\"speaker\":\"{speaker}\",\"model\":\"{model}\"}}"
                ),
                decode: RespKind::JsonBase64("audios"),
            }
        }
        Provider::OpenAI => {
            let (base, model) = e.base_and_model("https://api.openai.com", "tts-1");
            let voice = if e.voice.is_empty() { "alloy" } else { &e.voice };
            TtsRequest {
                url: format!("{base}/v1/audio/speech"),
                headers: vec![
                    ("Authorization".to_string(), format!("Bearer {}", e.key)),
                    ("Content-Type".to_string(), "application/json".to_string()),
                ],
                body: format!("{{\"model\":\"{model}\",\"input\":{t},\"voice\":\"{voice}\",\"response_format\":\"wav\"}}"),
                decode: RespKind::AudioBytes,
            }
        }
    }
}

/// Wrap raw little-endian S16 mono PCM in a 44-byte WAV header so the shared
/// `audio::decode` path handles it (ElevenLabs `pcm_24000` returns headerless).
fn wrap_pcm_wav(pcm: &[u8], rate: u32) -> Vec<u8> {
    let n = pcm.len() as u32;
    let mut w = Vec::with_capacity(44 + pcm.len());
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + n).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&rate.to_le_bytes());
    w.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    w.extend_from_slice(&2u16.to_le_bytes()); // block align
    w.extend_from_slice(&16u16.to_le_bytes()); // bits
    w.extend_from_slice(b"data");
    w.extend_from_slice(&n.to_le_bytes());
    w.extend_from_slice(pcm);
    w
}

/// Extract the base64 payload from a JSON body at `field`. Handles Sarvam's
/// `"audios":["..."]` (array) and Inworld's `"audioContent":"..."` (string) —
/// a minimal scan, no full parse (bodies are MBs of base64).
fn json_b64_field(body: &[u8], field: &str) -> Option<String> {
    let s = core::str::from_utf8(body).ok()?;
    let pat = format!("\"{field}\"");
    let i = s.find(&pat)? + pat.len();
    let rest = s[i..].trim_start().strip_prefix(':')?.trim_start();
    // Skip an opening array bracket + quote, or a bare quote.
    let rest = rest.strip_prefix('[').unwrap_or(rest).trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Synthesize `text` remotely → mono S16 PCM at [`crate::sound::tts::RATE`].
/// This is what `voice_say` calls when a remote TTS backend is configured; the
/// PCM feeds the same chunked `speech_pump` queue as the local path.
pub fn synth(e: &Endpoint, text: &str) -> Result<Vec<i16>, String> {
    let req = build_tts(e, text);
    let hdrs: Vec<(&str, &str)> = req.headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let resp = http::request("POST", &req.url, &hdrs, req.body.as_bytes(), HTTP_TIMEOUT_MS).map_err(|e| format!("remote tts: {e}"))?;
    let body = resp.body;
    if resp.status < 200 || resp.status >= 300 {
        let msg = core::str::from_utf8(&body).unwrap_or("").chars().take(200).collect::<String>();
        return Err(format!("remote tts: HTTP {} {}", resp.status, msg));
    }
    let audio_bytes: Vec<u8> = match req.decode {
        RespKind::AudioBytes => {
            // ElevenLabs pcm_24000 is headerless; others send WAV/MP3.
            if e.provider == Provider::ElevenLabs {
                wrap_pcm_wav(&body, 24000)
            } else {
                body
            }
        }
        RespKind::JsonBase64(field) => {
            let b64 = json_b64_field(&body, field).ok_or_else(|| format!("remote tts: no '{field}' in response"))?;
            crate::net::ws::base64_decode(&b64).ok_or("remote tts: bad base64 audio")?
        }
    };
    let audio = crate::audio::decode(&audio_bytes).map_err(|e| format!("remote tts: decode {e}"))?;
    let pcm = if audio.rate == crate::sound::tts::RATE {
        audio.pcm
    } else {
        crate::sound::resample(&audio.pcm, audio.rate, crate::sound::tts::RATE)
    };
    Ok(pcm)
}

// --- STT ---------------------------------------------------------------------

/// Build a `multipart/form-data` body uploading `wav` as file field `file`
/// plus the given text fields. Returns `(content_type_header, body)`.
pub fn multipart_wav(wav: &[u8], fields: &[(&str, &str)]) -> (String, Vec<u8>) {
    const BOUND: &str = "----ChittiVoiceBoundary7a1c";
    let mut b = Vec::new();
    for (k, v) in fields {
        b.extend_from_slice(format!("--{BOUND}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
    }
    b.extend_from_slice(
        format!("--{BOUND}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes(),
    );
    b.extend_from_slice(wav);
    b.extend_from_slice(format!("\r\n--{BOUND}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={BOUND}"), b)
}

/// STT endpoint URL + auth header + the transcript field to read, per provider.
fn stt_wire(e: &Endpoint) -> (String, (String, String), &'static [(&'static str, &'static str)], &'static str) {
    match e.provider {
        Provider::ElevenLabs => (
            "https://api.elevenlabs.io/v1/speech-to-text".to_string(),
            ("xi-api-key".to_string(), e.key.clone()),
            &[("model_id", "scribe_v1")],
            "text",
        ),
        Provider::Sarvam => (
            "https://api.sarvam.ai/speech-to-text".to_string(),
            ("api-subscription-key".to_string(), e.key.clone()),
            &[("model", "saarika:v2")],
            "transcript",
        ),
        // OpenAI-compatible whisper (also Groq); base via url@model.
        _ => (
            {
                let (base, _) = e.base_and_model("https://api.openai.com", "whisper-1");
                format!("{base}/v1/audio/transcriptions")
            },
            ("Authorization".to_string(), format!("Bearer {}", e.key)),
            &[("model", "whisper-1"), ("response_format", "json")],
            "text",
        ),
    }
}

/// Transcribe mono S16 `pcm` remotely → text. Called by `voice_stt_file` when
/// a remote STT backend is configured.
pub fn transcribe(e: &Endpoint, pcm: &[i16], rate: u32) -> Result<String, String> {
    // Bytes: reinterpret &[i16] as LE bytes, wrap as WAV.
    // SAFETY: reading i16 samples as their LE byte representation.
    let raw = unsafe { core::slice::from_raw_parts(pcm.as_ptr() as *const u8, pcm.len() * 2) };
    let wav = wrap_pcm_wav(raw, rate);
    let (url, auth, fields, field) = stt_wire(e);
    let (ctype, body) = multipart_wav(&wav, fields);
    let hdrs: Vec<(&str, &str)> = vec![(auth.0.as_str(), auth.1.as_str()), ("Content-Type", ctype.as_str())];
    let resp = http::request("POST", &url, &hdrs, &body, HTTP_TIMEOUT_MS).map_err(|e| format!("remote stt: {e}"))?;
    if resp.status < 200 || resp.status >= 300 {
        let msg = resp.text().chars().take(200).collect::<String>();
        return Err(format!("remote stt: HTTP {} {}", resp.status, msg));
    }
    let j = core::str::from_utf8(&resp.body).ok().and_then(Json::parse).ok_or("remote stt: non-JSON response")?;
    j.get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("remote stt: no '{field}' in response"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn config_roundtrip() {
        let cfg = VoiceConfig {
            tts: Some(Endpoint { provider: Provider::ElevenLabs, key: "sk-abc".into(), voice: "Rachel".into(), model: "".into() }),
            stt: Some(Endpoint { provider: Provider::OpenAI, key: "sk-o".into(), voice: "".into(), model: "http://lan:8000@whisper-1".into() }),
        };
        let mut obj = Vec::new();
        obj.push(("tts".to_string(), endpoint_json(cfg.tts.as_ref().unwrap())));
        obj.push(("stt".to_string(), endpoint_json(cfg.stt.as_ref().unwrap())));
        let bytes = Json::Obj(obj).to_pretty();
        let back = parse_config_json(bytes.as_bytes());
        assert_eq!(back.tts.as_ref().unwrap().provider, Provider::ElevenLabs);
        assert_eq!(back.tts.as_ref().unwrap().voice, "Rachel");
        assert_eq!(back.stt.as_ref().unwrap().provider, Provider::OpenAI);
        assert_eq!(back.stt.as_ref().unwrap().model, "http://lan:8000@whisper-1");
    }

    #[test_case]
    fn keyless_endpoint_is_dropped() {
        let j = Json::parse("{\"provider\":\"cartesia\",\"key\":\"\"}").unwrap();
        assert!(parse_endpoint(&j).is_none());
    }

    #[test_case]
    fn tts_request_shapes_per_provider() {
        let ep = |p| Endpoint { provider: p, key: "K".into(), voice: "".into(), model: "".into() };
        // Text is JSON-escaped, never string-concatenated raw.
        let r = build_tts(&ep(Provider::ElevenLabs), "hi \"there\"");
        assert!(r.url.contains("/text-to-speech/"));
        assert!(r.headers.iter().any(|(k, v)| k == "xi-api-key" && v == "K"));
        assert!(r.body.contains("\\\"there\\\""), "{}", r.body);
        assert_eq!(r.decode, RespKind::AudioBytes);

        let r = build_tts(&ep(Provider::Cartesia), "x");
        assert!(r.headers.iter().any(|(k, _)| k == "Cartesia-Version"));
        assert!(r.body.contains("pcm_s16le"));

        let r = build_tts(&ep(Provider::Inworld), "x");
        assert_eq!(r.decode, RespKind::JsonBase64("audioContent"));
        assert!(r.headers.iter().any(|(k, v)| k == "Authorization" && v.starts_with("Basic ")));

        let r = build_tts(&ep(Provider::Sarvam), "x");
        assert_eq!(r.decode, RespKind::JsonBase64("audios"));

        let r = build_tts(&ep(Provider::OpenAI), "x");
        assert!(r.url == "https://api.openai.com/v1/audio/speech");
    }

    #[test_case]
    fn openai_base_url_override() {
        let e = Endpoint { provider: Provider::OpenAI, key: "K".into(), voice: "".into(), model: "http://box:9000@myvoice".into() };
        let r = build_tts(&e, "x");
        assert_eq!(r.url, "http://box:9000/v1/audio/speech");
        assert!(r.body.contains("\"myvoice\""));
    }

    #[test_case]
    fn json_b64_field_extraction() {
        assert_eq!(json_b64_field(b"{\"audioContent\":\"QUJD\"}", "audioContent").unwrap(), "QUJD");
        assert_eq!(json_b64_field(b"{\"request_id\":\"x\",\"audios\":[\"WFla\"]}", "audios").unwrap(), "WFla");
        assert!(json_b64_field(b"{\"other\":1}", "audios").is_none());
    }

    #[test_case]
    fn multipart_has_boundary_and_file() {
        let (ct, body) = multipart_wav(b"RIFFxxxx", &[("model", "whisper-1")]);
        assert!(ct.starts_with("multipart/form-data; boundary="));
        let s = alloc::string::String::from_utf8_lossy(&body);
        assert!(s.contains("name=\"model\""));
        assert!(s.contains("filename=\"audio.wav\""));
        assert!(s.contains("RIFFxxxx"));
        assert!(s.trim_end().ends_with("--"));
    }

    #[test_case]
    fn pcm_wav_header_is_valid() {
        let w = wrap_pcm_wav(&[1, 2, 3, 4], 24000);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(&w[8..12], b"WAVE");
        assert_eq!(&w[44..], &[1, 2, 3, 4]);
    }
}
