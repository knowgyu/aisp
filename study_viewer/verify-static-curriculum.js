#!/usr/bin/env node
/* No-dependency verifier for the static Markdown study-note viewer. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { spawnSync } = require('child_process');

const ROOT = process.cwd();
const APP_DIR = path.join(ROOT, 'study_viewer');
let failures = 0;

function pass(message) { console.log(`PASS ${message}`); }
function fail(message) { failures += 1; console.error(`FAIL ${message}`); }
function assertCheck(condition, message) { condition ? pass(message) : fail(message); }
function readRequired(relativePath) {
  const absolutePath = path.join(ROOT, relativePath);
  assertCheck(fs.existsSync(absolutePath), `${relativePath} exists`);
  if (!fs.existsSync(absolutePath)) return '';
  const text = fs.readFileSync(absolutePath, 'utf8');
  assertCheck(text.trim().length > 0, `${relativePath} is non-empty`);
  console.log(`INFO ${relativePath} lines=${text.split('\n').length} chars=${text.length}`);
  return text;
}
function localImageRefs(markdown) {
  const refs = [];
  const pattern = /!\[[^\]]*\]\(([^)]+)\)/g;
  let match;
  while ((match = pattern.exec(markdown))) {
    const target = match[1].split('#')[0].trim();
    if (!target || /^(https?:|data:|#)/i.test(target)) continue;
    refs.push(target);
  }
  return refs;
}

function validateStaticFiles() {
  const html = readRequired('study_viewer/index.html');
  const styles = readRequired('study_viewer/styles.css');
  const render = readRequired('study_viewer/render.js');
  readRequired('study_viewer/notes-manifest.js');
  assertCheck(/marked@/.test(html), 'index.html loads a Markdown renderer');
  assertCheck(/mermaid@/.test(html), 'index.html loads Mermaid');
  assertCheck(/mathjax@/.test(html), 'index.html loads MathJax');
  assertCheck(/notes-manifest\.js/.test(html), 'index.html loads notes manifest');
  assertCheck(/render\.js/.test(html), 'index.html loads render.js');
  assertCheck(/AI_STUDY_NOTES/.test(render), 'render.js consumes AI_STUDY_NOTES');
  assertCheck(/fetch\(note\.path/.test(render), 'render.js fetches Markdown files');
  assertCheck(/Pretendard|Noto Sans KR/.test(styles), 'styles include requested Korean font stack');
  assertCheck(/color-scheme:\s*dark/.test(styles), 'styles default to dark color scheme');

  const jsFiles = fs.readdirSync(APP_DIR).filter((file) => file.endsWith('.js')).sort();
  for (const file of jsFiles) {
    const relativePath = `study_viewer/${file}`;
    const result = spawnSync(process.execPath, ['--check', relativePath], { cwd: ROOT, encoding: 'utf8' });
    assertCheck(result.status === 0, `${relativePath} passes node --check syntax`);
    if (result.status !== 0) console.error((result.stderr || result.stdout || '').trim());
  }
}

function loadManifest() {
  const context = { window: {} };
  vm.createContext(context);
  try {
    vm.runInContext(fs.readFileSync(path.join(APP_DIR, 'notes-manifest.js'), 'utf8'), context, { filename: 'notes-manifest.js' });
  } catch (error) {
    fail(`notes-manifest.js executes (${error.message})`);
    return [];
  }
  const notes = context.window.AI_STUDY_NOTES;
  assertCheck(Array.isArray(notes), 'window.AI_STUDY_NOTES is an array');
  assertCheck(notes.length >= 20, 'manifest includes the study note set');
  return Array.isArray(notes) ? notes : [];
}

function validateNotes(notes) {
  const ids = new Set();
  for (const [index, note] of notes.entries()) {
    const label = note && note.id ? note.id : `note[${index}]`;
    assertCheck(note && typeof note === 'object', `${label} is an object`);
    if (!note || typeof note !== 'object') continue;
    for (const key of ['id', 'title', 'section', 'path']) {
      assertCheck(typeof note[key] === 'string' && note[key].trim().length > 0, `${label} has ${key}`);
    }
    assertCheck(!ids.has(note.id), `${label} id is unique`);
    ids.add(note.id);
    assertCheck(note.path.startsWith('study_notes/'), `${label} path stays under study_notes/`);
    assertCheck(note.path.endsWith('.md'), `${label} path is Markdown`);
    const absolutePath = path.join(APP_DIR, note.path);
    assertCheck(fs.existsSync(absolutePath), `${label} Markdown file exists`);
    if (!fs.existsSync(absolutePath)) continue;
    const markdown = fs.readFileSync(absolutePath, 'utf8');
    assertCheck(markdown.trim().length > 0, `${label} Markdown is non-empty`);
    assertCheck((markdown.match(/```/g) || []).length % 2 === 0, `${label} code fences are balanced`);
    for (const ref of localImageRefs(markdown)) {
      const imagePath = path.resolve(path.dirname(absolutePath), ref);
      assertCheck(fs.existsSync(imagePath), `${label} image exists: ${ref}`);
    }
  }
  assertCheck(notes.some((note) => /on_device_ai/.test(note.path)), 'manifest includes On-Device AI notes');
  assertCheck(notes.some((note) => /language/.test(note.path)), 'manifest includes Language notes');
  assertCheck(notes.some((note) => /vision/.test(note.path)), 'manifest includes Vision notes');
}

validateStaticFiles();
validateNotes(loadManifest());

if (failures > 0) {
  console.error(`\nRESULT FAIL ${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nRESULT PASS static Markdown study-note viewer');
