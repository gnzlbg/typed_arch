#!/usr/bin/env bash

set -ex

: ${TARGET?"The TARGET environment variable must be set."}

rustup target add ${TARGET} || true

cargo test --target=${TARGET}
cargo test --release --target=${TARGET}
