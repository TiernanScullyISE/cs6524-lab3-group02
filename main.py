
from pathlib import Path

from mypylibs.features import track_orb


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for name in ('a', 'b'):
        track_orb(root / 'in' / f'{name}.mp4', root / 'out')
