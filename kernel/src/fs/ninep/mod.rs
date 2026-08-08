//! **9P2000.L** — the protocol behind a host shared folder.
//!
//! Split in two so the interesting half is testable:
//!
//! * [`wire`] — the message codec. Pure slice in, slice out.
//! * [`client`] — fid lifetime, path walking and the chunked read/write/readdir
//!   loops, over an abstract [`client::Rpc`] transport. Its unit tests run the
//!   whole client against an in-memory 9P server, which is where the fid leaks,
//!   the short-walk misreads and the chunk-resumption bugs actually surface.
//!
//! The device that carries it is [`crate::drivers::virtio_9p`].

pub mod client;
pub mod wire;

pub use client::{describe, Session};
pub use wire::{Attr, DirEntry9, P9Error, Qid};
