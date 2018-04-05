#!/usr/bin/env bash

set -ex

: ${TARGET?"The TARGET environment variable must be set."}

rustup target add ${TARGET} || true

cargo install cross

cross test --target=${TARGET}
cross test --release --target=${TARGET}
