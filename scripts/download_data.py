"""Download verified official inputs; never silently substitute Test for Slim."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = json.loads((ROOT/'data/sources.json').read_text())
    destination = ROOT/'data/raw'
    destination.mkdir(parents=True,exist_ok=True)
    for name, info in sources['files'].items():
        path = destination/name
        if not path.exists():
            request = urllib.request.Request(info['url'],headers={'User-Agent':'XWines-research/1.0'})
            with urllib.request.urlopen(request,timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != info['sha256']:
                raise ValueError(f'{name}: checksum mismatch. A Drive HTML page is not a CSV. '
                                 'Use the official folder link in data/sources.json and verify the file.')
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'{name}: checksum mismatch')
        print(f'Verified {name}')


if __name__ == '__main__':
    main()
