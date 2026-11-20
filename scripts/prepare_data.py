"""Prepare and verify the published splits without running model experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import prepare_splits

if __name__ == '__main__':
    for protocol, splits in prepare_splits(ROOT).items():
        print(protocol, {part: len(frame) for part, frame in splits.items()})
