"""Build the windowed dataset from toxicflow recordings.

Usage:
    PYTHONPATH=. .venv/bin/python examples/build_data.py \
        ~/workspace/toxicflow/data/chunk1 ... ~/workspace/toxicflow/data/chunk6

Writes data/book_windows.npz (gitignored; rebuild it yourself), plus
data/split_manifest.json (chunk hashes and session-disjoint split
assignment) and data/dataset_card.md (sessions, hours, splits).

Splits are session-disjoint: whole recording sessions go to train, val,
or test, never split across them. By default the first 70% of sessions
(chronological) go to train, next 15% to val, remainder to test. Pass
--split train:0,1,2,3,4 val:5 test:6,7 to lock an explicit manifest
(chunk indices are chronological).
"""

import argparse
import os

from flowforge.data import (build_dataset, dataset_card, save_dataset,
                            write_manifest)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "book_windows.npz")
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "data",
                        "split_manifest.json")
CARD = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_card.md")


def parse_split(spec: str | None) -> dict[str, list[int]] | None:
    if not spec:
        return None
    out: dict[str, list[int]] = {}
    for part in spec.split():
        name, idxs = part.split(":")
        out[name.strip()] = [int(i) for i in idxs.split(",") if i.strip()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", nargs="+", help="chunk directories")
    ap.add_argument("--split", default=None,
                    help='explicit manifest, e.g. "train:0,1,2,3,4 val:5 test:6,7"')
    args = ap.parse_args()
    ds = build_dataset(args.chunks, session_split=parse_split(args.split))
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    save_dataset(ds, os.path.abspath(OUT))
    manifest = write_manifest(ds, os.path.abspath(MANIFEST))
    with open(os.path.abspath(CARD), "w") as f:
        f.write(dataset_card(ds))
    for name in ("train", "val", "test"):
        s = getattr(ds, name)
        n_sess = len(set(s.session_ids.tolist()))
        print(f"{name}: {s.x.shape[0]} windows from {n_sess} sessions")
    print(f"seconds of book data: {ds.n_seconds}, chunks: {ds.n_chunks}")
    print(f"wrote {os.path.abspath(OUT)}")
    print(f"wrote {os.path.abspath(MANIFEST)}")
    print(f"wrote {os.path.abspath(CARD)}")
    print("split manifest:", manifest["session_split"])


if __name__ == "__main__":
    main()
