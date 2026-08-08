//! **SSH client** — RFC 4251/4252/4253/4254 over the kernel's TCP stack.
//!
//! Built so `/git clone git@host:repo` works, and so `/ssh` can run a command,
//! open a shell, and forward a port — which is the same machinery: SSH is one
//! encrypted transport carrying multiplexed channels, and a git fetch is a
//! channel running `git-upload-pack`.
//!
//! Layers, bottom up, each pure and unit-tested off the network:
//!
//! * [`wire`] — the data types (RFC 4251 §5) and binary packet (§6).
//! * [`kex`] — algorithm negotiation, the exchange hash, key derivation.
//! * [`cipher`] — packet encryption in all three length conventions.
//! * [`hostkey`] — host-key blobs, signature verification, `known_hosts`.
//! * [`auth`] — `publickey`/`password`, and reading an OpenSSH private key.
//! * [`channel`] — the connection protocol: sessions, exec, pty, forwards.
//!
//! [`client`] is the one part that touches the network: it orders the above and
//! does the I/O, bounded and pumping `upkeep` like every other blocking loop.
//!
//! Determinism boundary: all of this is native code *below* the boundary. The
//! model never implements a protocol; the `ssh` agent supplies identity and the
//! login/tunnel policy, and the bytes are produced here.

pub mod auth;
pub mod channel;
pub mod client;
pub mod cipher;
pub mod hostkey;
pub mod kex;
pub mod table;
pub mod wire;
