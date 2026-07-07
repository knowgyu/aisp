# Validation report

The current public study viewer serves `study_notes/` through a sidebar-based Markdown renderer.

Validated properties:

- static shell files exist and are non-empty
- JavaScript files pass `node --check`
- `notes-manifest.js` exposes `window.AI_STUDY_NOTES`
- every manifest Markdown file exists
- local image references inside Markdown resolve after copying `study_notes/`
- `index.html` includes Markdown, Mermaid, MathJax, manifest, and renderer scripts
- local HTTP smoke test can serve the app root and manifest

Run:

```bash
node study_viewer/verify-static-curriculum.js
```
