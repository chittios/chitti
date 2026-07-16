//! **Session** — the serializable, resumable unit of persistence
//! (`CHITTI_AGENTIC_HANDOFF.md` Phase A). A session holds the message history,
//! todo list, env/cwd, live capability set, skill scopes, sub-agent ledger, and
//! budget accounting for one conversation with the orchestrator.
//!
//! It deliberately does NOT hold sub-agent transcripts (isolation — only their
//! summaries cross back) nor the raw KV cache (recomputed on resume from the
//! seed + messages). That makes a session cheap to serialize and resume
//! deterministically.
//!
//! * [`session`] — construction + message/token bookkeeping on [`Session`].
//! * [`store`] — persist / resume / fork via the memory store (`synapse::fs`).
//! * [`todo`] — the todo list + `todo_write` semantics.

pub mod session;
pub mod jsonl;
pub mod store;
pub mod todo;

pub use store::{fork, list_summaries, resume, save, save_fork, search_sessions};
