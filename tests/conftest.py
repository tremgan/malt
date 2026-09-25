"""Loaded by pytest before any test module is collected."""

import os

# Sampling tests use nutpie, which needs PyTensor's C backend off on macOS 26+
# (see CLAUDE.md). This must run before anything imports pytensor, and any test
# module importing `malt.engine.glm` does; setting it here covers them all.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
