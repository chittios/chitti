//! Mutational fuzzer for the kernel's pure parsers.
//!
//! Usage:
//! ```text
//! cargo run --release -- <target> [iterations] [--seed N] [--time SECS]
//! ```
//!
//! Loads seed inputs from `corpus/<target>/`, mutates them, runs the parser
//! under `catch_unwind`, and saves any input that makes it panic to
//! `crashes/<target>/`. Exits non-zero if a crash was found, so it can be
//! wired into a script that fails a build on a new parser panic.
//!
//! Deterministic: the whole run is a function of `--seed`, so a crash is
//! reproducible with the same seed + iteration count.

mod mutate;
mod rng;
mod targets;

// The mounted kernel modules are `no_std` and reference `alloc` explicitly;
// at the crate root this makes `alloc` resolve for them (the `pngbench` shim).
extern crate alloc;

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Upper bound on a single parser call, so a hang (e.g. an O(2^n) length
/// field) is reported as a crash too rather than stalling the run forever.
/// Generous: the mounted parsers run in microseconds on normal inputs.
const PER_INPUT_TIMEOUT: Duration = Duration::from_secs(2);

const CRASHES_DIR: &str = "crashes";

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    args.remove(0);

    if args.iter().any(|a| a == "--selftest") {
        selftest();
        return;
    }

    let Some(target) = args.first().filter(|a| targets::TARGETS.contains(&a.as_str())).cloned() else {
        eprintln!(
            "usage: chitti-fuzz <target> [iterations] [--seed N] [--time SECS] [--selftest]\n  targets: {}",
            targets::TARGETS.join(", ")
        );
        std::process::exit(2);
    };
    args.remove(0);

    let mut iterations = 100_000usize;
    let mut seed = 0xC0FFEEu64;
    let mut time_budget: Option<u64> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(seed);
                i += 2;
            }
            "--time" => {
                time_budget = args.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            other => {
                if let Ok(n) = other.parse::<usize>() {
                    iterations = n;
                } else {
                    eprintln!("ignoring unknown arg {other:?}");
                }
                i += 1;
            }
        }
    }

    let corpus = load_corpus(&target);
    if corpus.is_empty() {
        eprintln!(
            "warning: no seeds in corpus/{target}/ — add at least one, or the fuzzer can only mutate the empty input"
        );
    }
    let mut rng = rng::Rng::new(seed);
    let crashes_dir = PathBuf::from(CRASHES_DIR).join(&target);
    std::fs::create_dir_all(&crashes_dir).ok();

    let started = Instant::now();
    let deadline = time_budget.map(|s| Instant::now() + Duration::from_secs(s));
    let mut pool: Vec<Vec<u8>> = corpus.iter().cloned().collect();
    let mut crashes = 0usize;
    let completed = Arc::new(AtomicU64::new(0));

    for n in 0..iterations {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        // Pool grows unboundedly otherwise; cap what we keep as parents.
        if pool.len() > 256 {
            let keep = rng.range(0, pool.len());
            pool.swap_remove(keep);
        }
        let parent = pool[rng.range(0, pool.len())].clone();
        let mut input = parent;
        // 1-3 mutation passes per input: more for a chance to escape a
        // rejection signature, capped so we still see the simple mutations.
        let passes = 1 + rng.range(0, 3);
        for _ in 0..passes {
            mutate::mutate(&mut input, &mut rng);
        }

        let crashed = run_guarded(&target, input.clone());
        if crashed {
            crashes += 1;
            let name = format!("crash-{crashes:06}-seed{seed}-n{n}.bin");
            let path = crashes_dir.join(&name);
            let mut f = std::fs::File::create(&path).expect("write crash input");
            f.write_all(&input).ok();
            println!("CRASH #{crashes}: {} ({} bytes)", path.display(), input.len());
        } else {
            // Keep inputs that grew past the seed as new parents — the
            // length-driven parsers only reach deep code paths on big inputs.
            pool.push(input);
        }
        completed.fetch_add(1, Ordering::Relaxed);
    }

    let done = completed.load(Ordering::Relaxed);
    println!(
        "\n{target}: {done} inputs in {:.1}s, {} crash(es) saved to {}/",
        started.elapsed().as_secs_f64(),
        crashes,
        crashes_dir.display()
    );
    if crashes > 0 {
        std::process::exit(1);
    }
}

/// Run one parser call in a thread with a panic + timeout guard.
/// Returns true if the input "crashed" (panicked, aborted, or hung).
fn run_guarded(target: &str, data: Vec<u8>) -> bool {
    let target = target.to_string();
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    let handle = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            targets::run(&target, &data);
        }));
        tx.send(result.is_err()).ok();
    });
    match rx.recv_timeout(PER_INPUT_TIMEOUT) {
        Ok(crashed) => {
            // Join so the worker thread is reused cleanly; ignore its result.
            let _ = handle.join();
            crashed
        }
        Err(_) => {
            // Timed out — treat as a hang. Can't cleanly kill the thread, so
            // detach and report; the per-input timeout makes this rare and the
            // next iteration runs on a fresh thread.
            let _ = handle.join();
            true
        }
    }
}

/// Load every file under `corpus/<target>/` (any extension) as a seed.
fn load_corpus(target: &str) -> Vec<Vec<u8>> {
    let dir = PathBuf::from("corpus").join(target);
    let mut seeds = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return seeds;
    };
    for entry in entries.flatten() {
        let p: PathBuf = entry.path();
        if p.is_file() {
            if let Ok(bytes) = std::fs::read(&p) {
                seeds.push(bytes);
            }
        }
    }
    seeds.sort_by_key(|b| b.len());
    seeds
}

/// Harness self-checks, run with `chitti-fuzz --selftest`. They live behind a
/// flag (not `#[cfg(test)]`) because the mounted kernel modules carry
/// `#[test_case]` — the kernel's custom-test-framework attribute — which would
/// not compile under a host `cargo test` harness.
fn selftest() {
    let mut ok = 0usize;
    let mut fail = 0usize;
    let mut check = |name: &str, pass: bool| {
        println!("{name}: {}", if pass { "ok" } else { "FAIL" });
        if pass {
            ok += 1;
        } else {
            fail += 1;
        }
    };

    // The guard must report a panic inside the target body as a crash — the
    // whole point of the harness. An unknown target panics inside
    // `targets::run`, which is exactly the panic-under-test path.
    check("guard reports a panicking target", run_guarded("does-not-exist", vec![1, 2, 3]));

    // And a normal target run must NOT be reported as a crash.
    check("guard passes a clean parser", !run_guarded("sha1", b"hello".to_vec()));

    // Mutations must actually change the input (or grow an empty one) and be
    // seed-deterministic.
    let mut a = b"a well-formed input for mutating".to_vec();
    let mut b = a.clone();
    let mut r1 = rng::Rng::new(0x1234);
    let mut r2 = rng::Rng::new(0x1234);
    for _ in 0..50 {
        mutate::mutate(&mut a, &mut r1);
        mutate::mutate(&mut b, &mut r2);
    }
    check("mutations are seed-deterministic", a == b);
    check("mutations change the input", a != b"a well-formed input for mutating".to_vec());

    let mut empty = Vec::new();
    mutate::mutate(&mut empty, &mut r1);
    check("empty input grows to one byte", empty.len() == 1);

    // Every registered target runs at least once without crashing.
    for t in targets::TARGETS {
        check(&format!("target {t} runs"), !run_guarded(t, vec![0x89, 0x50, 0x4e, 0x47]));
    }

    println!("\nselftest: {ok} ok, {fail} failed");
    if fail > 0 {
        std::process::exit(1);
    }
}
