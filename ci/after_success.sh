#!/bin/bash

# This runs only when a commit is pushed to master. It is responsible for
# updating docs and computing coverage statistics.

set -ex

if [ "$TRAVIS_RUST_VERSION" != "nightly" ] || [ "$TRAVIS_PULL_REQUEST" != "false" ] || [ "$TRAVIS_BRANCH" != "master" ]; then
  exit 0
fi

env

# Install kcov.
tmp=$(mktemp -d)
pushd "$tmp"

wget https://github.com/SimonKagstrom/kcov/archive/master.zip
unzip master.zip
mv kcov-master kcov
mkdir kcov/build
current=$(pwd)
cd kcov/build
cmake ..
make
export PATH="$(pwd)/src:${PATH}"
cd ${current}

popd
./ci/run-kcov --coveralls-id $TRAVIS_JOB_ID
