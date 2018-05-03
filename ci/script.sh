#!/bin/sh

# This is the main CI script for testing the remake crate

set -ex

# Builds the regex crate and runs tests.
cargo build --verbose
cargo doc --verbose

# Run tests.
cargo test --verbose
