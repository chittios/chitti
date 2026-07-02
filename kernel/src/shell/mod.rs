//! The **intent shell** (`CHITTI_OS_HANDOFF.md` Phase 5): the serial console
//! where a human types an intent and an agent carries it out. It is the
//! top-of-stack entry point for the whole system -- a line of text becomes a
//! plan, which becomes capability-checked, audited Synapse calls, which
//! become effects.
//!
//! Two entry points: [`run_intent`] executes a single intent with a fresh
//! agent and returns the result (used by the boot demo and the test suite),
//! and [`run`] is the interactive read-eval loop over COM1 that a person
//! drives at `cargo xtask run`.

use crate::persona::{self, Agent, Planner, RulePlanner};
use crate::serial;
use crate::{serial_print, serial_println};
use alloc::string::{String, ToString};

/// Route one intent to a fresh, general-purpose agent, run its plan to
/// completion, and return the final result. The agent's *live* context is
/// fresh each call, but the persistent memory store (tier 2) is global, so
/// facts remembered by an earlier intent are recallable here -- which is what
/// makes the "recall a fact not in live context" behaviour observable.
pub fn run_intent(intent: &str) -> String {
    let mut agent = Agent::spawn(persona::default_manifest("shell-agent"));
    let mut planner = RulePlanner;
    agent.begin(intent, &mut planner);
    let result = agent.run_to_completion().to_string();
    agent.kill();
    result
}

/// Boot-time demonstration of the Phase 5 deliverable (fast + deterministic,
/// so it runs on every boot before the interactive shell): a typed intent
/// completing a multi-primitive plan, and a fact recalled from the persistent
/// store by a *different* agent than the one that stored it.
pub fn demo() {
    serial_println!("Chitti: --- Persona + intent shell (Phase 5) ---");
    serial_println!("Chitti: planner = {}", RulePlanner.name());

    let intent = "write a file called notes with the text hello world, then read it back";
    serial_println!("Chitti: intent> {}", intent);
    serial_println!("Chitti: result< {}", run_intent(intent));

    // Store a fact with one agent; recall it with another (its live context
    // never contained it -- it comes from the persistent store, tier 2).
    serial_println!("Chitti: intent> remember that project is chitti");
    serial_println!("Chitti: result< {}", run_intent("remember that project is chitti"));
    serial_println!("Chitti: intent> what is project");
    serial_println!("Chitti: result< {}", run_intent("what is project"));
}

/// The interactive intent shell: read a line from COM1, run it as an intent
/// (or a builtin), print the result, repeat. Never returns -- it is the
/// system's steady state. A single session agent is reused so live context
/// (and memory) carries across intents within a session.
pub fn run() -> ! {
    serial_println!("");
    serial_println!("Chitti: intent shell ready.");
    serial_println!("  Type an intent, e.g.: write a file called todo with the text buy milk, then read it back");
    serial_println!("  Builtins: help | infer | exit");

    let mut agent = Agent::spawn(persona::default_manifest("chitti"));
    let mut planner = RulePlanner;
    let mut line = String::new();

    loop {
        serial_print!("chitti> ");
        line.clear();
        read_line(&mut line);
        let intent = line.trim();
        if intent.is_empty() {
            continue;
        }
        match intent {
            "exit" | "quit" => {
                serial_println!("Chitti: shell exiting; system idle.");
                loop {
                    crate::arch::x86_64::hlt();
                }
            }
            "help" => {
                serial_println!("  intents: write a file called X with the text Y[, then read it back]");
                serial_println!("           remember that K is V | what is K | list | say TEXT");
                serial_println!("  builtins: infer (run the Cortex reference inference), exit");
            }
            "infer" => run_infer(),
            _ => {
                agent.begin(intent, &mut planner);
                let result = agent.run_to_completion();
                serial_println!("=> {}", result);
            }
        }
    }
}

/// Builtin: run the Cortex reference inference (Phase 3) on demand from the
/// shell. Slow under QEMU TCG, hence not on the automatic boot path.
fn run_infer() {
    match crate::cortex::run_reference_inference() {
        Some(r) => serial_println!("=> (tokens={:?}, matches reference={})", r.continuation, r.matched_reference),
        None => serial_println!("=> no model module present; boot with the model bundled to use `infer`"),
    }
}

/// Read a line from COM1 into `buf`, echoing as it goes and handling
/// backspace. Cooperatively yields the CPU while no input is available, so
/// other tasks keep running while the shell waits at the prompt.
fn read_line(buf: &mut String) {
    loop {
        match serial::read_byte() {
            Some(b'\r') | Some(b'\n') => {
                serial_println!("");
                return;
            }
            Some(0x7f) | Some(0x08) => {
                if buf.pop().is_some() {
                    // Erase the character on the terminal: back up, overwrite
                    // with a space, back up again.
                    serial::put_byte(0x08);
                    serial::put_byte(b' ');
                    serial::put_byte(0x08);
                }
            }
            Some(c @ 0x20..=0x7e) => {
                buf.push(c as char);
                serial::put_byte(c);
            }
            Some(_) => {} // ignore other control bytes
            None => crate::sched::yield_now(),
        }
    }
}
