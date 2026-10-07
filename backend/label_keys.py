"""Write data/keys.json from the K: lines of ABC files (e.g. Nottingham).

Each tune in an ABC file has an X: number; the MIDI file for it is expected
to be named <abc file name><X number>.mid (the Nottingham convention, e.g.
ashover.abc tune X:1 -> ashover1.mid). Tunes in a mode other than major or
minor are skipped, and get a detected key during training instead.

    python label_keys.py path/to/nottingham-dataset/ABC_cleaned
"""
import argparse
import glob
import json
import os
import re

MAJOR_WORDS = {'', 'maj', 'major', 'ion', 'ionian'}
MINOR_WORDS = {'m', 'min', 'minor', 'aeo', 'aeolian'}


def parse_key_line(line: str):
    """[key name, 'major'|'minor'] from an ABC K: line, or None (other modes)"""
    k = re.match(r'K:\s*([A-G][b#]?)\s*([A-Za-z]*)', line)
    if not k:
        return None
    word = k.group(2).lower()
    if word in MAJOR_WORDS:
        return [k.group(1), 'major']
    if word in MINOR_WORDS:
        return [k.group(1), 'minor']
    return None


def label_folder(abc_folder: str) -> dict:
    labels = {}
    for path in sorted(glob.glob(os.path.join(abc_folder, '*.abc'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        tune = None
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                x = re.match(r'X:\s*(\d+)', line)
                if x:
                    tune = x.group(1)
                    continue
                if line.startswith('K:') and tune is not None:
                    name = f'{stem}{tune}.mid'
                    label = parse_key_line(line)
                    if label and name not in labels:
                        labels[name] = label
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('abc_folder')
    parser.add_argument('--out', default='data/keys.json')
    args = parser.parse_args()

    labels = label_folder(args.abc_folder)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=0)
    minor = sum(1 for _, mode in labels.values() if mode == 'minor')
    print(f"Wrote {len(labels)} labels ({minor} minor) to {args.out}")
