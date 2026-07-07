# Verification

The viewer is verified as a no-build static site.

## Commands

```bash
for f in study_viewer/*.js; do node --check "$f"; done
node study_viewer/verify-static-curriculum.js
python3 -m http.server 4173 --directory study_viewer >/tmp/ai-study-notes-http.log 2>&1 &
server_pid=$!
curl -fsSI --max-time 5 http://127.0.0.1:4173/
curl -fsS --max-time 5 http://127.0.0.1:4173/notes-manifest.js >/dev/null
kill "$server_pid"
```

Expected result: JavaScript syntax checks pass, the verifier reports `RESULT PASS`, and the local HTTP smoke returns 200.
