# aisp

## Study viewer build

Static viewer artifacts live under `study_viewer/` and are checked in for GitHub Pages.
For local rebuilds, prefer the incremental builder:

```bash
python scripts/build_study_viewer.py --verify
```

It skips notebook HTML conversion when the source `.ipynb` and
`scripts/convert_ipynb_static.py` are older than the existing HTML, then syncs
only changed Markdown/assets into `study_viewer/`.

Use a full notebook regeneration only after broad converter/layout changes:

```bash
python scripts/build_study_viewer.py --force --verify
```
