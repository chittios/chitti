//! Pure helpers for the audio-player HUD and the status-bar now-playing chip.
//!
//! Kept out of `framebuffer/` so the formatters, playlist arithmetic and the
//! spectrum analyser are unit-tested (the compositor is `cfg(not(test))`).

use alloc::string::String;
use alloc::vec::Vec;

/// Bars in the live spectrum. Enough to read as an analyser; cheap to paint.
pub const SPECTRUM_BARS: usize = 40;

/// How far a peak-hold bar falls per refresh (~4 Hz from the audio tab).
pub const SPECTRUM_DECAY: u8 = 20;

/// Signature of the **static** chrome (track + playlist flags). Live fields
/// (clock, spectrum, play/pause) are not in it — a 4 Hz tick that only
/// changes those must not rebuild the whole pane.
pub fn chrome_sig(path: &str, idx: usize, n: usize, repeat: Repeat, shuffle: bool) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in path.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    for x in [idx as u64, n as u64, repeat as u64, shuffle as u64] {
        h ^= x;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Repeat mode for the playlist. `One` only affects auto-advance (user
/// next/prev still step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Repeat {
    Off,
    All,
    One,
}

impl Repeat {
    pub fn cycle(self) -> Self {
        match self {
            Repeat::Off => Repeat::All,
            Repeat::All => Repeat::One,
            Repeat::One => Repeat::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Repeat::Off => "Off",
            Repeat::All => "All",
            Repeat::One => "One",
        }
    }
}

/// `mm:ss` with zero-padded minutes, the clock the player and chip share.
pub fn format_mmss(ms: u64) -> String {
    let total = ms / 1000;
    alloc::format!("{:02}:{:02}", total / 60, total % 60)
}

/// Filename stem for the HUD / chip: strip the directory, drop a known audio
/// extension, turn underscores into spaces. `sample.wav` → `sample`.
pub fn display_title(path: &str) -> String {
    let base = path.rsplit('/').next().unwrap_or(path);
    let stem = match base.rfind('.') {
        Some(i) if is_audio_ext(&base[i + 1..]) => &base[..i],
        _ => base,
    };
    if stem.is_empty() {
        return String::from(base);
    }
    stem.replace('_', " ")
}

/// True when `name` (a file name or a path) is something `/open` will play.
pub fn is_audio_filename(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    match base.rfind('.') {
        Some(i) => is_audio_ext(&base[i + 1..]),
        None => false,
    }
}

fn is_audio_ext(ext: &str) -> bool {
    let mut buf = [0u8; 4];
    let n = ext.len().min(4);
    for (i, b) in ext.as_bytes().iter().take(n).enumerate() {
        buf[i] = b.to_ascii_lowercase();
    }
    matches!(&buf[..n], b"wav" | b"mp3" | b"aac" | b"m4a")
}

/// Parent directory of `path`. `"/a/b.wav"` → `"/a"`; `"/b.wav"` → `"/"`.
pub fn parent_dir(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) => "/",
        Some(i) => &path[..i],
        None => ".",
    }
}

/// Join `parent` and a file name. Does not normalise `.` / `..`.
pub fn join_dir(parent: &str, name: &str) -> String {
    if parent == "/" {
        alloc::format!("/{name}")
    } else if parent.is_empty() || parent == "." {
        String::from(name)
    } else {
        alloc::format!("{parent}/{name}")
    }
}

/// Keep names that look like audio files, sorted, and always include `current`
/// even if the directory listing missed it (a store path with no siblings).
pub fn playlist_from_names(current: &str, names: &[&str]) -> Vec<String> {
    let parent = parent_dir(current);
    let mut out: Vec<String> = names
        .iter()
        .copied()
        .filter(|n| is_audio_filename(n))
        .map(|n| {
            if n.contains('/') {
                String::from(n)
            } else {
                join_dir(parent, n)
            }
        })
        .collect();
    if !out.iter().any(|p| p == current) {
        out.push(String::from(current));
    }
    out.sort();
    out.dedup();
    out
}

/// Visible slice of a playlist around `idx`, at most `rows` entries.
pub fn playlist_window(idx: usize, len: usize, rows: usize) -> (usize, usize) {
    if len == 0 || rows == 0 {
        return (0, 0);
    }
    if len <= rows {
        return (0, len);
    }
    let half = rows / 2;
    let start = idx.saturating_sub(half).min(len - rows);
    (start, start + rows)
}

/// Next index after the current track ends. `One` stays put; `All` wraps;
/// `Off` yields `None` past the last track. Shuffle picks a different slot
/// from `seed` (the wall clock is a fine seed).
pub fn auto_next(
    cur: usize,
    len: usize,
    repeat: Repeat,
    shuffle: bool,
    seed: u64,
) -> Option<usize> {
    if len == 0 {
        return None;
    }
    match repeat {
        Repeat::One => Some(cur.min(len - 1)),
        Repeat::Off => {
            if shuffle {
                shuffle_other(cur, len, seed)
            } else if cur + 1 < len {
                Some(cur + 1)
            } else {
                None
            }
        }
        Repeat::All => {
            if shuffle {
                shuffle_other(cur, len, seed).or(Some(cur))
            } else {
                Some((cur + 1) % len)
            }
        }
    }
}

/// User next/prev: always steps, wraps at the ends. Shuffle still hops.
pub fn user_step(cur: usize, len: usize, shuffle: bool, seed: u64, forward: bool) -> usize {
    if len == 0 {
        return 0;
    }
    if len == 1 {
        return 0;
    }
    if shuffle {
        return shuffle_other(cur, len, seed).unwrap_or(cur);
    }
    if forward {
        (cur + 1) % len
    } else if cur == 0 {
        len - 1
    } else {
        cur - 1
    }
}

fn shuffle_other(cur: usize, len: usize, seed: u64) -> Option<usize> {
    if len < 2 {
        return None;
    }
    // Anything but `cur`. The +1 bias means seed 0 still moves.
    Some((cur + 1 + (seed as usize % (len - 1))) % len)
}

/// Status-bar chip. Empty when nothing is loaded so the template swallows
/// the separator (same posture as `${notifications}` / `${recording}`).
///
/// Playing → play icon + title; paused / finished → pause icon + title.
pub fn chip_text(loaded: bool, playing: bool, title: &str) -> String {
    if !loaded {
        return String::new();
    }
    let icon = if playing {
        crate::icons::fa::PLAY
    } else {
        crate::icons::fa::PAUSE
    };
    let name = crate::textsel::ellipsize(&display_title(title), 22);
    alloc::format!("{icon} {name}")
}

/// Octave-band energies of a mono window, bass on the left, `0..=255` per bar.
///
/// Each step is a 2-tap Haar split: the high-pass half is one octave (treble
/// first), then we recurse on the low-pass residual. No trig, so it stays
/// cheap on every 4 Hz refresh and is easy to pin with a DC vs Nyquist pair.
pub fn spectrum_bands(mono: &[i16], n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    if mono.is_empty() {
        return alloc::vec![0u8; n];
    }
    let mut lo: Vec<i32> = mono.iter().map(|&s| s as i32).collect();
    let mut bands: Vec<u64> = Vec::new();
    while lo.len() >= 2 && bands.len() < 12 {
        let mut next = Vec::with_capacity(lo.len() / 2);
        let mut acc: u64 = 0;
        let mut count: u64 = 0;
        for c in lo.chunks_exact(2) {
            next.push((c[0] + c[1]) / 2);
            let h = (c[0] - c[1]) / 2;
            acc += (h as i64).unsigned_abs();
            count += 1;
        }
        bands.push(if count == 0 { 0 } else { acc / count });
        lo = next;
    }
    if !lo.is_empty() {
        let acc: u64 = lo.iter().map(|&x| (x as i64).unsigned_abs()).sum();
        bands.push(acc / lo.len() as u64);
    }
    // Collected treble-first; the analyser reads bass → treble left to right.
    bands.reverse();
    resample_u8(&bands, n)
}

/// Peak-hold: each bar is `max(now, prev - drop)` so a transient lingers
/// instead of collapsing in one refresh.
pub fn decay_spectrum(prev: &[u8], now: &[u8], drop: u8) -> Vec<u8> {
    let n = now.len();
    let mut out = alloc::vec![0u8; n];
    for i in 0..n {
        let held = prev.get(i).copied().unwrap_or(0).saturating_sub(drop);
        out[i] = held.max(now.get(i).copied().unwrap_or(0));
    }
    out
}

fn resample_u8(src: &[u64], n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    if src.is_empty() {
        return alloc::vec![0u8; n];
    }
    let max = src.iter().copied().max().unwrap_or(0).max(1);
    let last = src.len() - 1;
    let mut out = alloc::vec![0u8; n];
    for i in 0..n {
        let pos = if n == 1 {
            0
        } else {
            i * last / (n - 1)
        };
        let e = src[pos];
        let v = ((e * 255) / max).min(255) as u8;
        out[i] = if e > 0 { v.max(1) } else { 0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn format_mmss_zero_pads_minutes() {
        assert_eq!(format_mmss(0), "00:00");
        assert_eq!(format_mmss(3_000), "00:03");
        assert_eq!(format_mmss(209_000), "03:29");
        assert_eq!(format_mmss(3_661_000), "61:01");
    }

    #[test_case]
    fn display_title_strips_dir_ext_and_underscores() {
        assert_eq!(display_title("/samples/audios/sample.wav"), "sample");
        assert_eq!(display_title("Sub_Urban-Cradles.mp3"), "Sub Urban-Cradles");
        assert_eq!(display_title("readme.txt"), "readme.txt");
        assert_eq!(display_title("noext"), "noext");
    }

    #[test_case]
    fn is_audio_filename_accepts_the_openable_set() {
        assert!(is_audio_filename("a.wav"));
        assert!(is_audio_filename("B.MP3"));
        assert!(is_audio_filename("/x/y.aac"));
        assert!(is_audio_filename("clip.m4a"));
        assert!(!is_audio_filename("clip.mp4"));
        assert!(!is_audio_filename("notes.md"));
        assert!(!is_audio_filename("wav"));
    }

    #[test_case]
    fn parent_and_join_round_trip() {
        assert_eq!(parent_dir("/samples/audios/a.wav"), "/samples/audios");
        assert_eq!(parent_dir("/a.wav"), "/");
        assert_eq!(parent_dir("rel.wav"), ".");
        assert_eq!(join_dir("/samples/audios", "b.mp3"), "/samples/audios/b.mp3");
        assert_eq!(join_dir("/", "x.wav"), "/x.wav");
    }

    #[test_case]
    fn playlist_from_names_filters_sorts_and_keeps_current() {
        let p = playlist_from_names(
            "/m/now.wav",
            &["z.mp3", "notes.md", "a.wav", "now.wav"],
        );
        assert_eq!(
            p,
            alloc::vec![
                String::from("/m/a.wav"),
                String::from("/m/now.wav"),
                String::from("/m/z.mp3"),
            ]
        );
        let solo = playlist_from_names("/m/only.aac", &["readme.txt"]);
        assert_eq!(solo, alloc::vec![String::from("/m/only.aac")]);
    }

    #[test_case]
    fn playlist_window_keeps_current_in_view() {
        assert_eq!(playlist_window(0, 3, 9), (0, 3));
        assert_eq!(playlist_window(5, 9, 5), (3, 8));
        assert_eq!(playlist_window(0, 9, 5), (0, 5));
        assert_eq!(playlist_window(8, 9, 5), (4, 9));
        assert_eq!(playlist_window(0, 0, 5), (0, 0));
    }

    #[test_case]
    fn auto_next_respects_repeat_and_shuffle() {
        assert_eq!(auto_next(0, 3, Repeat::Off, false, 0), Some(1));
        assert_eq!(auto_next(2, 3, Repeat::Off, false, 0), None);
        assert_eq!(auto_next(2, 3, Repeat::All, false, 0), Some(0));
        assert_eq!(auto_next(1, 3, Repeat::One, false, 0), Some(1));
        assert_eq!(auto_next(0, 1, Repeat::Off, false, 0), None);
        assert_eq!(auto_next(0, 1, Repeat::All, false, 0), Some(0));
        let hop = auto_next(1, 4, Repeat::All, true, 7).unwrap();
        assert_ne!(hop, 1);
        assert!(hop < 4);
    }

    #[test_case]
    fn user_step_wraps_and_shuffle_moves() {
        assert_eq!(user_step(0, 3, false, 0, true), 1);
        assert_eq!(user_step(2, 3, false, 0, true), 0);
        assert_eq!(user_step(0, 3, false, 0, false), 2);
        assert_eq!(user_step(0, 1, false, 0, true), 0);
        let hop = user_step(2, 5, true, 3, true);
        assert_ne!(hop, 2);
    }

    #[test_case]
    fn chip_text_is_empty_when_idle() {
        assert_eq!(chip_text(false, true, "x.wav"), "");
        let play = chip_text(true, true, "/m/Cradles.mp3");
        assert!(play.contains("Cradles"), "{play}");
        assert!(play.starts_with(crate::icons::fa::PLAY), "{play}");
        let pause = chip_text(true, false, "Cradles.mp3");
        assert!(pause.starts_with(crate::icons::fa::PAUSE), "{pause}");
        // Empty chip drops the following separator, so a quiet machine stays
        // byte-identical to one whose template never had the variable.
        let resolve = |v: &str| -> String {
            match v {
                "nowplaying" => chip_text(false, false, ""),
                "datetime_short" => String::from("19:19"),
                _ => String::new(),
            }
        };
        assert_eq!(
            crate::ui_config::expand("${nowplaying}  ${datetime_short}", &resolve),
            "19:19"
        );
    }

    #[test_case]
    fn spectrum_silence_is_flat_dc_is_bass_nyquist_is_treble() {
        assert!(spectrum_bands(&[], 8).iter().all(|&b| b == 0));
        assert!(spectrum_bands(&[0; 256], 8).iter().all(|&b| b == 0));
        let dc = alloc::vec![20_000i16; 256];
        let low = spectrum_bands(&dc, 8);
        let left: u32 = low[..4].iter().map(|&x| x as u32).sum();
        let right: u32 = low[4..].iter().map(|&x| x as u32).sum();
        assert!(left > right, "DC should sit on the left: {low:?}");
        let mut nyq = alloc::vec![0i16; 256];
        for (i, s) in nyq.iter_mut().enumerate() {
            *s = if i % 2 == 0 { 20_000 } else { -20_000 };
        }
        let high = spectrum_bands(&nyq, 8);
        let left: u32 = high[..4].iter().map(|&x| x as u32).sum();
        let right: u32 = high[4..].iter().map(|&x| x as u32).sum();
        assert!(right > left, "Nyquist should sit on the right: {high:?}");
    }

    #[test_case]
    fn decay_spectrum_holds_then_falls() {
        let prev = [0u8, 100, 10];
        let now = [40u8, 20, 10];
        let out = decay_spectrum(&prev, &now, 20);
        assert_eq!(out, alloc::vec![40, 80, 10]);
    }

    #[test_case]
    fn repeat_cycles_off_all_one() {
        assert_eq!(Repeat::Off.cycle(), Repeat::All);
        assert_eq!(Repeat::All.cycle(), Repeat::One);
        assert_eq!(Repeat::One.cycle(), Repeat::Off);
        assert_eq!(Repeat::All.label(), "All");
    }

    #[test_case]
    fn chrome_sig_changes_only_with_chrome() {
        let a = chrome_sig("/m/a.wav", 0, 2, Repeat::Off, false);
        assert_eq!(a, chrome_sig("/m/a.wav", 0, 2, Repeat::Off, false));
        assert_ne!(a, chrome_sig("/m/b.wav", 0, 2, Repeat::Off, false));
        assert_ne!(a, chrome_sig("/m/a.wav", 1, 2, Repeat::Off, false));
        assert_ne!(a, chrome_sig("/m/a.wav", 0, 2, Repeat::All, false));
        assert_ne!(a, chrome_sig("/m/a.wav", 0, 2, Repeat::Off, true));
    }
}
