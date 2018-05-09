#!/bin/sh

# This is the main CI script for testing the remake crate

set -ex

# Builds the regex crate and docs
cargo build --verbose
cargo doc --verbose

# Check style.
#
# Only run the check on nightly because write-mode checking
# is not yet stable.
if [ "$TRAVIS_RUST_VERSION" == "nightly" ]; then
    rustup component add rustfmt-preview
    cargo fmt --all -- --write-mode=check
fi

# Run tests.
cargo test --verbose
