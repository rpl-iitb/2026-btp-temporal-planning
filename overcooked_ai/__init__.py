"""Benchmark 1: Overcooked-AI 1.1.0 (pinned simulator source in repo/)."""
import os
import warnings

# Set before any submodule imports overcooked_ai_py (it pulls in pygame / gym).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
