#!/bin/sh
set -eu

cd "$(dirname "$0")"

if [ -x ./target/release/ingexuity-server ]; then
    exec ./target/release/ingexuity-server
fi

exec cargo run --release -p ingexuity-server
