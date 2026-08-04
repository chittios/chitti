//! Analog clock-face geometry: where a hand or an hour mark points, and how long
//! and thick it is. Pure integer math — no framebuffer, no float, no `sqrt`.
//!
//! It lives here rather than beside the painter because
//! [`crate::framebuffer`] is `#[cfg(not(test))]` (see the gate in `lib.rs`), so
//! **anything inside the compositor cannot be unit-tested at all** — a
//! `#[cfg(test)] mod` in there is never even compiled. This is the same split
//! `editor_wrap` makes for the editor's soft-wrap math, and it is not academic:
//! the quadrant reduction below was wrong in two of four quadrants and shipped,
//! because the only thing that would have caught it was a test that could run.
//!
//! Convention: **0° is 12 o'clock and angles increase clockwise**, and the
//! returned offsets are screen-space, where `+y` is **down**.

/// `cos(deg)` in Q10 (1024 = 1.0) over the first quadrant, indexed 0..=90.
/// `round(1024 * cos(d°))`, so `COS_Q10[0] == 1024` and `COS_Q10[90] == 0`
/// exactly — the ends being exact is what lets the reduction below be pure
/// reflection with no special cases.
const COS_Q10: [i32; 91] = [
    1024, 1024, 1023, 1023, 1022, 1020, 1018, 1016, 1014, 1011, 1008, 1005, 1002, 998, 994, 989,
    984, 979, 974, 968, 962, 956, 949, 943, 935, 928, 920, 912, 904, 896, 887, 878, 868, 859, 849,
    839, 828, 818, 807, 796, 784, 773, 761, 749, 737, 724, 711, 698, 685, 672, 658, 644, 630, 616,
    602, 587, 573, 558, 543, 527, 512, 496, 481, 465, 449, 433, 416, 400, 384, 367, 350, 333, 316,
    299, 282, 265, 248, 230, 213, 195, 178, 160, 143, 125, 107, 89, 71, 54, 36, 18, 0,
];

/// cos/sin of `deg` (any integer, reduced mod 360) in Q10 (1024 = 1.0).
///
/// The quadrant reduction is the whole substance of this function, and it was
/// wrong for two of the four: quadrants II and III had the cos and sin
/// expressions **swapped**. That is why every hand and hour mark in the bottom
/// half of the dial pointed somewhere else, and why the second hand appeared to
/// run backwards from the 30 s mark on — 30 s is 180°, the first angle the broken
/// arm answers. The identities, spelled out so a future edit can be checked
/// against them rather than guessed:
///
/// - `cos(d) = -cos(180-d)` and `sin(d) =  cos(d-90)`  for `d` in 90..=180
/// - `cos(d) = -cos(d-180)` and `sin(d) = -cos(270-d)` for `d` in 180..=270
/// - `cos(d) =  cos(360-d)` and `sin(d) = -cos(d-270)` for `d` in 270..360
pub fn cos_sin_q10(deg: i32) -> (i32, i32) {
    let d = deg.rem_euclid(360) as usize;
    let cos_q = |a: usize| COS_Q10[a.min(90)];
    match d {
        0..=90 => (cos_q(d), cos_q(90 - d)),
        91..=180 => (-cos_q(180 - d), cos_q(d - 90)),
        181..=270 => (-cos_q(d - 180), -cos_q(270 - d)),
        _ => (cos_q(360 - d), -cos_q(d - 270)),
    }
}

/// `(dx, dy)` from the dial centre at `deg` clockwise from 12 o'clock, length `r`.
/// A negative `r` points the other way, which is how a hand's counterweight tail
/// is placed without a second angle.
pub fn clock_offset(r: i64, deg: i32) -> (i64, i64) {
    // Math angle from +x is 90° − deg, so cos/sin swap roles:
    // x = r·sin(deg) = r·cos(m), y = −r·cos(deg) = −r·sin(m).
    let m = (90 - deg).rem_euclid(360);
    let (c, s) = cos_sin_q10(m);
    ((r * c as i64) / 1024, -((r * s as i64) / 1024))
}

/// One hand's geometry, derived from the face radius: how far past the pivot the
/// tip reaches, the counterweight tail behind it, and the stroke width in
/// **tenths of a pixel** (so a 1.2 px second hand is expressible without floats).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HandGeom {
    pub len: i64,
    pub tail: i64,
    pub w10: i64,
}

/// `(hour, minute, second)` hand geometry for a face of radius `r`. The minimums
/// matter: at the smallest face the dropdown draws (`r == 28`) the ratios alone
/// would round the second hand's width below a pixel and it would vanish.
pub fn hand_geoms(r: i64) -> [HandGeom; 3] {
    [
        // Hour: short and thick, tucked inside the numerals.
        HandGeom { len: r * 52 / 100, tail: (r * 14 / 100).max(2), w10: (r * 10 / 11).max(30) },
        // Minute: reaches the tick ring, a touch narrower.
        HandGeom { len: r * 76 / 100, tail: (r * 15 / 100).max(2), w10: (r * 10 / 16).max(24) },
        // Second: thin, long, with the longest counterweight.
        HandGeom { len: r * 88 / 100, tail: (r * 20 / 100).max(3), w10: (r * 10 / 40).max(10) },
    ]
}

/// `(hour, minute, second)` hand angles in degrees clockwise from 12, for a local
/// wall time. The sub-steps are the point: an hour hand that only moved on the
/// hour, or a minute hand that only moved on the minute, is what makes a rendered
/// clock look broken even when every angle is individually right — so the hour
/// hand advances 1° every 2 minutes and the minute hand 1° every 10 seconds.
pub fn hand_angles(h: i64, mi: i64, s: i64) -> (i32, i32, i32) {
    (
        (h.rem_euclid(12) * 30 + mi / 2) as i32,
        (mi * 6 + s / 10) as i32,
        (s * 6) as i32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is the reference the quadrant reduction is checked against, so
    /// pin its ends and the exact angles (30/45/60) that a previous hand-written
    /// table disagreed with badly enough to need a `match` arm patching them.
    #[test_case]
    fn cosine_table_is_exact_at_the_named_angles() {
        assert_eq!(COS_Q10[0], 1024);
        assert_eq!(COS_Q10[30], 887);
        assert_eq!(COS_Q10[45], 724);
        assert_eq!(COS_Q10[60], 512);
        assert_eq!(COS_Q10[90], 0);
        // Monotonically decreasing across the quadrant — a transcription slip
        // shows up here even where no named angle pins the value.
        for d in 1..=90usize {
            assert!(COS_Q10[d] <= COS_Q10[d - 1], "COS_Q10 rises at {d}");
        }
    }

    /// cos/sin must satisfy the reflection identities in **every** quadrant.
    /// Quadrants II and III had the two expressions swapped, which is the whole
    /// bug: `cos_sin_q10(180)` answered `(0, 1024)` instead of `(-1024, 0)`.
    #[test_case]
    fn quadrants_agree_with_the_reflection_identities() {
        assert_eq!(cos_sin_q10(0), (1024, 0));
        assert_eq!(cos_sin_q10(90), (0, 1024));
        assert_eq!(cos_sin_q10(180), (-1024, 0));
        assert_eq!(cos_sin_q10(270), (0, -1024));
        for d in 0..360i32 {
            let (c, s) = cos_sin_q10(d);
            // cos(180-d) = -cos(d), sin(180-d) = sin(d)
            assert_eq!(cos_sin_q10(180 - d), (-c, s), "reflection about 90° fails at {d}");
            // cos(-d) = cos(d), sin(-d) = -sin(d)
            assert_eq!(cos_sin_q10(-d), (c, -s), "reflection about 0° fails at {d}");
            // On the unit circle, within Q10 rounding.
            let mag = (c * c + s * s) / 1024;
            assert!((mag - 1024).abs() <= 2, "|(cos,sin)| off at {d}: {mag}");
        }
    }

    /// Screen convention: 0° is 12 o'clock and `+y` is **down**.
    #[test_case]
    fn clock_offset_points_at_the_right_hour_mark() {
        assert_eq!(clock_offset(100, 0), (0, -100)); // 12
        assert_eq!(clock_offset(100, 90), (100, 0)); // 3
        assert_eq!(clock_offset(100, 180), (0, 100)); // 6
        assert_eq!(clock_offset(100, 270), (-100, 0)); // 9
        // A negative length is the counterweight tail: 12 o'clock, backwards.
        assert_eq!(clock_offset(-20, 0), (0, 20));
    }

    /// The regression this module was written for: past the 30 s mark the second
    /// hand ran backwards. Clockwise motion in screen coordinates (`y` down) means
    /// the cross product of consecutive tip vectors is **positive** at every step
    /// of the sweep — including the wrap from 59 s round to 0 s.
    #[test_case]
    fn second_hand_sweeps_clockwise_through_every_second() {
        let r = 100;
        for s in 0..60i32 {
            let (x0, y0) = clock_offset(r, s * 6);
            let (x1, y1) = clock_offset(r, (s + 1) % 60 * 6);
            let cross = x0 * y1 - y0 * x1;
            assert!(cross > 0, "hand goes backwards from {s}s: ({x0},{y0}) -> ({x1},{y1})");
        }
        // And the bottom half really is the bottom half (30 s used to point left).
        assert!(clock_offset(r, 30 * 6).1 > 0, "30s must point down");
        assert!(clock_offset(r, 45 * 6).0 < 0, "45s must point left");
    }

    /// Hour marks share the reduction with the hands, so a quadrant slip stacked
    /// whole hours on top of each other. All 12 must be distinct and each must sit
    /// in the quadrant its hour belongs to.
    #[test_case]
    fn twelve_hour_marks_are_distinct_and_in_quadrant() {
        let r = 100;
        let marks: [(i64, i64); 12] = core::array::from_fn(|h| clock_offset(r, h as i32 * 30));
        for a in 0..12 {
            for b in (a + 1)..12 {
                assert!(marks[a] != marks[b], "hour {a} and {b} land on {:?}", marks[a]);
            }
        }
        assert!(marks[1].0 > 0 && marks[1].1 < 0, "1 o'clock is up-right");
        assert!(marks[4].0 > 0 && marks[4].1 > 0, "4 o'clock is down-right");
        assert!(marks[7].0 < 0 && marks[7].1 > 0, "7 o'clock is down-left");
        assert!(marks[10].0 < 0 && marks[10].1 < 0, "10 o'clock is up-left");
    }

    /// The hands stay ordered short-thick to long-thin, and none of them rounds
    /// away or overruns the rim at any face size the dropdown draws.
    #[test_case]
    fn hand_proportions_hold_at_every_face_size() {
        for r in [28i64, 36, 48, 64, 96, 160] {
            let [h, m, s] = hand_geoms(r);
            assert!(h.len < m.len && m.len < s.len, "r={r} lengths {h:?} {m:?} {s:?}");
            assert!(h.w10 > m.w10 && m.w10 > s.w10, "r={r} widths {h:?} {m:?} {s:?}");
            for g in [h, m, s] {
                // A hand spans `-tail ..= +len` about the pivot, so its reach is
                // `len` — the tail only has to stay a counterweight and not read
                // as a second, backwards hand.
                assert!(g.len < r, "r={r} hand tip leaves the dial: {g:?}");
                assert!(g.tail < g.len, "r={r} tail is as long as the hand: {g:?}");
                assert!(g.tail > 0 && g.w10 >= 10, "r={r} hand vanishes: {g:?}");
            }
        }
    }

    /// A hand that only jumps at its own unit reads as a broken clock, so pin the
    /// sub-steps: the hour hand creeps with the minutes, the minute hand with the
    /// seconds, and both complete exactly one turn.
    #[test_case]
    fn hands_advance_smoothly_and_wrap_once_per_turn() {
        assert_eq!(hand_angles(0, 0, 0), (0, 0, 0));
        assert_eq!(hand_angles(3, 0, 0), (90, 0, 0));
        assert_eq!(hand_angles(12, 0, 0), (0, 0, 0)); // noon and midnight agree
        assert_eq!(hand_angles(15, 0, 0), (90, 0, 0)); // 24-hour input
        // Half past: the hour hand sits between two hour marks, not on one.
        assert_eq!(hand_angles(1, 30, 0), (45, 180, 0));
        // The minute hand moves within the minute; the second hand does not
        // move within the second.
        assert_eq!(hand_angles(0, 0, 30), (0, 3, 180));
        // Monotonic through the whole 12-hour cycle, and one turn each.
        let mut prev = (0, 0, 0);
        for t in 1..12 * 60 * 60i64 {
            let (h, mi, s) = (t / 3600, t / 60 % 60, t % 60);
            let a = hand_angles(h, mi, s);
            assert!(a.0 >= prev.0, "hour hand went backwards at {t}s: {a:?} < {prev:?}");
            assert!(a.0 < 360 && a.1 < 360 && a.2 < 360, "angle out of turn at {t}s: {a:?}");
            prev = a;
        }
        assert_eq!(prev.0, 359, "hour hand should just about close the circle");
    }
}
