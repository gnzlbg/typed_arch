#!/usr/bin/env bash

set -ex

: ${TARGET?"The TARGET environment variable must be set."}

CARGO_CMD=cargo

if [[ ${TARGET} == *"linux"* ]]; then
    cargo install cross
    CARGO_CMD=cross
fi

${CARGO_CMD} test --target=${TARGET}
${CARGO_CMD} test --release --target=${TARGET}
