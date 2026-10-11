"""Build the windowed dataset from toxicflow recordings.

Usage:
    PYTHONPATH=. .venv/bin/python examples/build_data.py \
        ~/workspace/toxicflow/data/chunk1 ... ~/workspace/toxicflow/data/chunk6

Writes data/book_windows.npz (gitignored; rebuild it yourself). The split is
by time: first 70% train, next 15% val, last 15% test. No window content is
ever shared between splits.
"""

import os
import sys

from flowforge.data import build_dataset, save_dataset

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "book_windows.npz")


def main() -> None:
    chunk_dirs = sys.argv[1:]
    if not chunk_dirs:
        raise SystemExit("pass chunk directories, e.g. ~/workspace/toxicflow/data/chunk1 ...")
    ds = build_dataset(chunk_dirs)
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    save_dataset(ds, os.path.abspath(OUT))
    for name in ("train", "val", "test"):
        s = getattr(ds, name)
        print(f"{name}: {s.x.shape[0]} windows")
    print(f"seconds of book data: {ds.n_seconds}, chunks: {ds.n_chunks}")
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
