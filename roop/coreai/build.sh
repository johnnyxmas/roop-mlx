#!/bin/sh
# Rebuild libcoreai_bridge.dylib from CoreAIBridge.swift.
#
# The prebuilt dylib is committed, so users never need this.  Rebuilding
# needs the Xcode 27+ SDK: the CommandLineTools SDK has no CoreAI module.
# DEVELOPER_DIR selects Xcode for this build only (no xcode-select/sudo).
set -eu
cd "$(dirname "$0")"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
xcrun --sdk macosx swiftc -O -emit-library \
    -target arm64-apple-macos27.0 \
    CoreAIBridge.swift -o libcoreai_bridge.dylib
echo "built $(pwd)/libcoreai_bridge.dylib"
