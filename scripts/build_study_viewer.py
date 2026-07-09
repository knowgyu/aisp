#!/usr/bin/env python3
"""Incrementally build the static AISP study viewer.

This keeps the checked-in static artifact workflow, but avoids re-converting
notebooks whose source ipynb and converter are older than the generated HTML.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import sync_study_viewer

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / 'scripts' / 'convert_ipynb_static.py'
VERIFY = ROOT / 'study_viewer' / 'verify-static-curriculum.js'


def notebook_entries():
    return [note for note in sync_study_viewer.NOTES if note.get('kind') == 'notebook']


def is_stale(src: Path, out: Path) -> bool:
    if not out.exists():
        return True
    out_mtime = out.stat().st_mtime
    return src.stat().st_mtime > out_mtime or CONVERTER.stat().st_mtime > out_mtime


def convert_notebooks(*, force: bool = False) -> dict[str, int]:
    stats = {'converted': 0, 'skipped': 0}
    for note in notebook_entries():
        src = ROOT / note['sourceIpynb']
        out = ROOT / 'study_viewer' / note['notebookHtml']
        if not src.exists():
            raise FileNotFoundError(src)
        if not force and not is_stale(src, out):
            stats['skipped'] += 1
            continue
        subprocess.run(
            [sys.executable, str(CONVERTER), str(src), str(out), note['title']],
            cwd=ROOT,
            check=True,
        )
        stats['converted'] += 1
    return stats


def run_verify() -> None:
    subprocess.run(['node', str(VERIFY)], cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Incrementally build study_viewer artifacts')
    parser.add_argument('--force', action='store_true', help='reconvert every notebook')
    parser.add_argument('--verify', action='store_true', help='run the static curriculum verifier after build')
    args = parser.parse_args(argv)

    converted = convert_notebooks(force=args.force)
    copied = sync_study_viewer.copy_allowed()
    manifest_changed = sync_study_viewer.write_manifest()
    print(
        'build study_viewer: '
        f"converted={converted['converted']} skipped={converted['skipped']} "
        f"copied={copied['copied']} removed={copied['removed']} "
        f"manifest_changed={int(manifest_changed)}",
        flush=True,
    )
    if args.verify:
        run_verify()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
