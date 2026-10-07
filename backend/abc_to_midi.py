"""Convert tunes from ABC files into MIDI files plus key labels for training.

Needs music21, which pins newer numpy than torch does here, so install it in
a separate virtual environment from the one used for training:

    py -3.11 -m venv abcvenv
    abcvenv\Scripts\pip install music21
    abcvenv\Scripts\python abc_to_midi.py path/to/oneills1850/*.abc --prefix oneills

Each tune becomes data/midi_files/<prefix>_<abc file>_<X number>.mid, and its
key (from the K: line) is merged into data/keys.json. Tunes in modes other
than major or minor, and tunes music21 cannot parse, are skipped.
"""
import argparse
import glob
import json
import os
import re

from music21 import converter

from label_keys import parse_key_line


# Folk tunes are tens of bars; anything far longer is a mis-parse
MAX_QUARTER_NOTES = 400


def split_tunes(text: str):
    """Split an ABC file into one string per tune (each starts with X:)"""
    return [t for t in re.split(r'(?m)^(?=X:)', text) if t.startswith('X:')]


def tune_number(tune: str) -> str:
    return re.match(r'X:\s*(\d+)', tune).group(1)


def strip_inline_meters(tune: str) -> str:
    """Drop M: lines after the first; music21 chokes on repeated meters"""
    seen, lines = False, []
    for line in tune.splitlines():
        if line.startswith('M:'):
            if seen:
                continue
            seen = True
        lines.append(line)
    return '\n'.join(lines)


def to_stream(tune: str):
    try:
        return converter.parse(tune, format='abc')
    except Exception:
        return converter.parse(strip_inline_meters(tune), format='abc')


def remove_grace_notes(stream):
    """Grace notes have no duration; as MIDI they become stray short notes"""
    for note in list(stream.recurse().notes):
        if note.duration.isGrace:
            note.activeSite.remove(note)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('abc_files', nargs='+', help="ABC files (globs allowed)")
    parser.add_argument('--prefix', required=True,
                        help="prepended to output names to keep sources apart")
    parser.add_argument('--out', default='data/midi_files')
    parser.add_argument('--keys', default='data/keys.json')
    args = parser.parse_args()

    paths = sorted(p for pattern in args.abc_files for p in glob.glob(pattern))
    os.makedirs(args.out, exist_ok=True)
    labels = {}
    if os.path.exists(args.keys):
        with open(args.keys, encoding='utf-8') as f:
            labels = json.load(f)

    stats = {'written': 0, 'minor': 0, 'other_mode': 0, 'failed': 0}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding='utf-8', errors='ignore') as f:
            tunes = split_tunes(f.read())

        for tune in tunes:
            key_line = next((l for l in tune.splitlines() if l.startswith('K:')), None)
            label = parse_key_line(key_line) if key_line else None
            if label is None:
                stats['other_mode'] += 1
                continue

            name = f'{args.prefix}_{stem}_{tune_number(tune)}.mid'
            try:
                stream = to_stream(tune)
                remove_grace_notes(stream)
                if stream.highestTime > MAX_QUARTER_NOTES:
                    raise ValueError('implausibly long')
                stream.write('midi', fp=os.path.join(args.out, name))
            except Exception:
                stats['failed'] += 1
                continue

            labels[name] = label
            stats['written'] += 1
            stats['minor'] += label[1] == 'minor'
        print(f"{os.path.basename(path)}: {stats}", flush=True)

    with open(args.keys, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=0)
    print(f"Done: {stats}; {len(labels)} labels in {args.keys}")


if __name__ == '__main__':
    main()
