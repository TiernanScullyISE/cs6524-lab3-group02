
from pathlib import Path

import yaml

from mypylibs.comparison import compare_features
from mypylibs.features import track_features
from mypylibs.sfm import load_camera, extract_views, estimate_pairs


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    method = compare_features(root / 'in', root / 'out')
    for name in ('a', 'b'):
        track_features(root / 'in' / f'{name}.mp4', root / 'out', method=method)
        K, D = load_camera(root / 'in')
        config = yaml.safe_load((root / 'in' / f'{name}.yaml').read_text())
        views = extract_views(root / 'in' / f'{name}.mp4', K, D, config, method)
        estimate_pairs(views, K, root / 'out' / f'{name}_10u', config.get('min_motion', 0))
