
from pathlib import Path

from mypylibs.comparison import compare_features
from mypylibs.features import track_features


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    method = compare_features(root / 'in', root / 'out')
    for name in ('a', 'b'):
        track_features(root / 'in' / f'{name}.mp4', root / 'out', method=method)
