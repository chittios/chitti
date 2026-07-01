#!/bin/sh
# Invoked by cargo as the `[target.x86_64-chitti] runner` (see
# kernel/.cargo/config.toml) with the path to a compiled test binary.
#
# We `cd` into xtask/ before calling `cargo run`: cargo's own config-file
# discovery walks up from the *current working directory*, not from
# --manifest-path. Cargo test's runner subprocess inherits kernel/'s cwd,
# which carries kernel/.cargo/config.toml's custom target/build-std/
# rustflags settings — those would otherwise leak into building xtask
# itself (a normal host binary) and break it.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
exec cargo run --quiet -- runner "$@"
