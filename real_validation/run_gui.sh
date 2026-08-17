#!/usr/bin/env sh
cd "$(dirname "$0")/.." || exit 1
exec python -m real_validation.main
