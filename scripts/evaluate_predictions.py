"""Score a RatingID,prediction CSV against a verified evaluation partition."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import PROTOCOLS
from src.pipeline import load_prepared_splits, score_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--protocol', choices=PROTOCOLS, default='chronological')
    parser.add_argument('--partition', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    _, splits = load_prepared_splits(ROOT, args.protocol)
    result = score_predictions(splits[args.partition], pd.read_csv(args.predictions))
    print(json.dumps(result, indent=2))
