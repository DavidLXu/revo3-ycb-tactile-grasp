"""Asset locations (the repository ships everything the demo needs under ./assets)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets'
FINGERTIPS = ASSETS / 'revo3' / 'fingertips'
HAND = ASSETS / 'revo3' / 'hand'
YCB = ASSETS / 'ycb'
OBJECT_LABELS = {
    '013_apple': 'Apple',
    '017_orange': 'Orange',
    '011_banana': 'Banana',
    '005_tomato_soup_can': 'Tomato soup can',
    '025_mug': 'Mug',
    '006_mustard_bottle': 'Mustard bottle',
}
OBJECTS = tuple(OBJECT_LABELS)
