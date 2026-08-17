//! **Per-agent accounting** — what each agent has actually consumed.
//!
//! `/top` reports tasks; this reports *agents*, which is the unit a human cares
//! about on this OS. An agent is not a task (several agents share the shell's
//! task, and a delegation spawns its own), so scheduler ticks alone cannot
//! answer "which agent burnt the afternoon".
//!
//! It is also the missing prerequisite for per-agent quotas: you cannot cap what
//! you do not measure. The counters here are deliberately the ones a quota would
//! be written against — model tokens, wall time, primitive invocations, bytes
//! written — rather than a general-purpose metrics system.
//!
//! ## Counting rules that are easy to get wrong
//!
//! **A refused call still costs.** A denied primitive ran the whole gate chain
//! and, more importantly, the model produced the tokens that asked for it. An
//! agent that spends its budget generating refused calls has still spent it, and
//! a quota that only counted successes would never stop a runaway.
//!
//! **Time is wall time, not CPU time.** An agent blocked on a network read is not
//! burning CPU, but it is holding a slot and the human is waiting. Both numbers
//! are useful; this records the one the human perceives, and the scheduler
//! already has the other.
//!
//! **Delegation is charged to the child and summarised to the parent**, so a
//! parent's own line stays readable while `total_with_children` still answers
//! "what did this request cost".

use crate::mm::Locked;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// One agent's running totals.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Usage {
    /// Prompt tokens fed to the model.
    pub prompt_tokens: u64,
    /// Tokens the model generated.
    pub output_tokens: u64,
    /// Wall milliseconds spent in this agent's turns.
    pub wall_ms: u64,
    /// Primitive invocations that reached the executor, refused or not.
    pub calls: u64,
    /// Of those, how many the gate chain refused.
    pub refused: u64,
    /// Bytes written to the store.
    pub bytes_written: u64,
    /// Completed turns.
    pub turns: u64,
}

impl Usage {
    /// Total tokens, the number a cost estimate is usually keyed on.
    pub fn tokens(&self) -> u64 {
        self.prompt_tokens + self.output_tokens
    }

    /// Output tokens per second over this agent's wall time — the figure that
    /// tells you whether an agent is thinking or stuck.
    pub fn tokens_per_sec(&self) -> f32 {
        if self.wall_ms == 0 {
            return 0.0;
        }
        (self.output_tokens as f32) * 1000.0 / (self.wall_ms as f32)
    }

    pub fn merge(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.output_tokens += other.output_tokens;
        self.wall_ms += other.wall_ms;
        self.calls += other.calls;
        self.refused += other.refused;
        self.bytes_written += other.bytes_written;
        self.turns += other.turns;
    }
}

#[derive(Default)]
struct Ledger {
    per_agent: BTreeMap<u64, Usage>,
    /// child -> parent, so a delegation's cost can roll up.
    parent: BTreeMap<u64, u64>,
}

static LEDGER: Locked<Ledger> =
    Locked::new(Ledger { per_agent: BTreeMap::new(), parent: BTreeMap::new() });

fn with<R>(f: impl FnOnce(&mut Ledger) -> R) -> R {
    LEDGER.with(f)
}

/// Record that `child` was delegated to by `parent`.
pub fn link(child: u64, parent: u64) {
    with(|l| {
        l.parent.insert(child, parent);
    });
}

pub fn add_prompt_tokens(agent: u64, n: u64) {
    with(|l| l.per_agent.entry(agent).or_default().prompt_tokens += n);
}

pub fn add_output_tokens(agent: u64, n: u64) {
    with(|l| l.per_agent.entry(agent).or_default().output_tokens += n);
}

pub fn add_wall_ms(agent: u64, ms: u64) {
    with(|l| l.per_agent.entry(agent).or_default().wall_ms += ms);
}

/// One primitive invocation. `refused` covers every non-executed outcome —
/// denied, tainted, out of scope, malformed — because all of them cost the
/// tokens that produced the call.
pub fn add_call(agent: u64, refused: bool) {
    with(|l| {
        let u = l.per_agent.entry(agent).or_default();
        u.calls += 1;
        if refused {
            u.refused += 1;
        }
    });
}

pub fn add_bytes_written(agent: u64, n: u64) {
    with(|l| l.per_agent.entry(agent).or_default().bytes_written += n);
}

pub fn add_turn(agent: u64) {
    with(|l| l.per_agent.entry(agent).or_default().turns += 1);
}

pub fn get(agent: u64) -> Usage {
    with(|l| l.per_agent.get(&agent).copied().unwrap_or_default())
}

/// This agent's usage plus every descendant's, for "what did that request cost".
///
/// Walks children rather than recursing on parents so a cycle — which should be
/// impossible, but a bad `link` would create one — cannot hang the shell.
pub fn total_with_children(agent: u64) -> Usage {
    with(|l| {
        let mut total = l.per_agent.get(&agent).copied().unwrap_or_default();
        let mut frontier = alloc::vec![agent];
        let mut seen = alloc::vec![agent];
        while let Some(cur) = frontier.pop() {
            for (&child, &parent) in l.parent.iter() {
                if parent == cur && !seen.contains(&child) {
                    seen.push(child);
                    frontier.push(child);
                    if let Some(u) = l.per_agent.get(&child) {
                        total.merge(u);
                    }
                }
            }
        }
        total
    })
}

/// Every agent with recorded usage, heaviest by tokens first.
pub fn ranked() -> Vec<(u64, Usage)> {
    with(|l| {
        let mut v: Vec<(u64, Usage)> = l.per_agent.iter().map(|(&k, &u)| (k, u)).collect();
        v.sort_by(|a, b| b.1.tokens().cmp(&a.1.tokens()).then_with(|| a.0.cmp(&b.0)));
        v
    })
}

/// A `/top`-style table.
pub fn report() -> String {
    let rows = ranked();
    if rows.is_empty() {
        return "no agent activity recorded".into();
    }
    let mut s = String::from("agent    tokens    in/out          calls  refused   wrote  tok/s\n");
    for (id, u) in rows {
        s.push_str(&alloc::format!(
            "{:<8} {:<9} {:>6}/{:<7} {:<6} {:<8} {:<7} {:.1}\n",
            id,
            u.tokens(),
            u.prompt_tokens,
            u.output_tokens,
            u.calls,
            u.refused,
            u.bytes_written,
            u.tokens_per_sec(),
        ));
    }
    s
}

pub fn reset() {
    with(|l| {
        l.per_agent.clear();
        l.parent.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn counters_accumulate_per_agent() {
        reset();
        add_prompt_tokens(1, 100);
        add_output_tokens(1, 40);
        add_prompt_tokens(2, 7);
        assert_eq!(get(1).tokens(), 140);
        assert_eq!(get(2).tokens(), 7);
        assert_eq!(get(99), Usage::default(), "an unknown agent reads as zero, not an error");
    }

    #[test_case]
    fn a_refused_call_still_counts_as_a_call() {
        // A quota that only counted successes would never stop an agent burning
        // its budget on calls the gate chain refuses.
        reset();
        add_call(1, false);
        add_call(1, true);
        let u = get(1);
        assert_eq!((u.calls, u.refused), (2, 1));
    }

    #[test_case]
    fn delegation_rolls_up_to_the_parent() {
        reset();
        add_output_tokens(1, 10);
        add_output_tokens(2, 25);
        link(2, 1);
        assert_eq!(get(1).tokens(), 10, "the parent's own line stays its own");
        assert_eq!(total_with_children(1).tokens(), 35);
    }

    #[test_case]
    fn rollup_handles_a_chain_of_delegations() {
        reset();
        add_output_tokens(1, 1);
        add_output_tokens(2, 2);
        add_output_tokens(3, 4);
        link(2, 1);
        link(3, 2);
        assert_eq!(total_with_children(1).tokens(), 7);
    }

    #[test_case]
    fn a_cycle_in_the_parent_map_does_not_hang() {
        // Should be impossible, but a hang here freezes the shell with
        // interrupts disabled -- worth making structurally safe.
        reset();
        add_output_tokens(1, 1);
        add_output_tokens(2, 2);
        link(2, 1);
        link(1, 2);
        assert_eq!(total_with_children(1).tokens(), 3);
    }

    #[test_case]
    fn ranking_puts_the_heaviest_agent_first() {
        reset();
        add_output_tokens(5, 10);
        add_output_tokens(6, 900);
        add_output_tokens(7, 100);
        let r = ranked();
        assert_eq!(r[0].0, 6);
        assert_eq!(r[1].0, 7);
    }

    #[test_case]
    fn tokens_per_second_is_zero_rather_than_infinite_at_zero_time() {
        reset();
        add_output_tokens(1, 50);
        assert_eq!(get(1).tokens_per_sec(), 0.0, "no divide-by-zero in a status line");
        add_wall_ms(1, 1000);
        assert!((get(1).tokens_per_sec() - 50.0).abs() < 0.01);
    }

    #[test_case]
    fn the_report_is_readable_when_empty() {
        reset();
        assert!(report().contains("no agent activity"));
    }
}
