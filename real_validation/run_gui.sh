#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1
exec python main_validation.py
