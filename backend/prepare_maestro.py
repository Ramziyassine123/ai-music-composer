"""Select Romantic piano pieces from MAESTRO and label their keys for training.

MAESTRO (CC BY-NC-SA 4.0, https://magenta.tensorflow.org/datasets/maestro) is
~1200 recorded piano performances. Download "maestro-v3.0.0-midi.zip", unzip
it, and run:

    python prepare_maestro.py path/to/maestro-v3.0.0

Each selected performance is copied to data/midi_files/maestro_<name>.midi and
its key, mode and style ("classical") are merged into data/keys.json. The key
comes from the piece's title ("Ballade No. 1 in G Minor, Op. 23"), so pieces
whose title names no single key (sets of preludes or etudes, variations) are
skipped. Titles give a piece's home key only: a piece that modulates a lot is
labelled with its home key throughout.
"""
import argparse
import csv
import json
import os
import re
import shutil

COMPOSERS = [
    'Frédéric Chopin', 'Franz Schubert', 'Franz Liszt', 'Robert Schumann',
    'Johannes Brahms', 'Sergei Rachmaninoff', 'Felix Mendelssohn',
    'Alexander Scriabin', 'Edvard Grieg', 'Pyotr Ilyich Tchaikovsky',
]

# A key letter, an optional accidental ("sharp", "-flat", "#", "b"), then the mode
KEY_RE = re.compile(
    r"\b([A-G])(?:[- ]?((?i:sharp|flat))|([#♯♭])|b(?=[ -]?(?i:major|minor)))?"
    r"[ -]?((?i:major|minor))\b")


def title_key(title: str):
    """[key name, 'major'|'minor'] if the title names exactly one key, else None"""
    keys = set()
    for m in KEY_RE.finditer(title):
        letter, word, symbol, mode = m.groups()
        accidental = ''
        if word:
            accidental = '#' if word.lower() == 'sharp' else 'b'
        elif symbol:
            accidental = '#' if symbol == '♯' or symbol == '#' else 'b'
        elif m.group(0)[1:2] == 'b':
            accidental = 'b'
        keys.add((letter + accidental, mode.lower()))
    if len(keys) != 1:
        return None
    key, mode = keys.pop()
    return [key, mode]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('maestro_dir', help="unzipped maestro-v3.0.0 folder")
    parser.add_argument('--out', default='data/midi_files')
    parser.add_argument('--keys', default='data/keys.json')
    parser.add_argument('--min-seconds', type=float, default=45,
                        help="skip performances shorter than this")
    args = parser.parse_args()

    with open(os.path.join(args.maestro_dir, 'maestro-v3.0.0.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    os.makedirs(args.out, exist_ok=True)
    labels = {}
    if os.path.exists(args.keys):
        with open(args.keys, encoding='utf-8') as f:
            labels = json.load(f)

    kept = skipped_key = skipped_short = 0
    minor = 0
    for row in rows:
        if row['canonical_composer'] not in COMPOSERS:
            continue
        label = title_key(row['canonical_title'])
        if label is None:
            skipped_key += 1
            continue
        if float(row['duration']) < args.min_seconds:
            skipped_short += 1
            continue

        name = 'maestro_' + os.path.basename(row['midi_filename'])
        shutil.copyfile(os.path.join(args.maestro_dir, row['midi_filename']),
                        os.path.join(args.out, name))
        labels[name] = label + ['classical']
        kept += 1
        minor += label[1] == 'minor'

    with open(args.keys, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=0)
    print(f"Kept {kept} performances ({minor} minor); skipped {skipped_key} with no single "
          f"key in the title and {skipped_short} that were too short. "
          f"{len(labels)} labels in {args.keys}")


if __name__ == '__main__':
    main()
