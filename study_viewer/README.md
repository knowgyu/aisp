# AISP Study Notes static viewer

No-build GitHub Pages viewer for the Markdown notes under `study_notes/`.

## Local preview

```bash
python3 -m http.server 4173 --directory study_viewer
# open http://127.0.0.1:4173/
```

Direct `file://` opening is not supported because the viewer fetches Markdown files.

## Files

- `index.html` — static shell and CDN dependencies for Markdown, Mermaid, and MathJax.
- `render.js` — sidebar routing, Markdown fetch/render, relative image rewriting, Mermaid/MathJax activation.
- `styles.css` — dark document-first reading UI using Pretendard/Noto Sans fallback.
- `notes-manifest.js` — generated list of Markdown notes.
- `study_notes/` — public copy of the study notes and image assets.
- `verify-static-curriculum.js` — no-dependency smoke verifier for the static viewer.

## Verification

```bash
for f in study_viewer/*.js; do node --check "$f"; done
node study_viewer/verify-static-curriculum.js
python3 -m http.server 4173 --directory study_viewer
```
