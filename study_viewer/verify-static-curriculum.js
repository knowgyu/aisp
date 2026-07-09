#!/usr/bin/env node
/* No-dependency verifier for the static AISP study-note viewer. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { spawnSync } = require('child_process');

const ROOT = process.cwd();
const APP_DIR = path.join(ROOT, 'study_viewer');
const DENIED_PUBLIC_PATHS = [
  'study_viewer/study_notes/on_device_ai/ch01_pdf_extracted_text.md',
  'study_viewer/study_notes/on_device_ai/ch01_lecture_pack.md',
  'study_viewer/study_notes/on_device_ai/ch01_lecture_pack.html',
  'study_viewer/study_notes/on_device_ai/01_cnn_pruning_deep_review.md'
];
const EXPECTED_NOTEBOOKS = new Map([
  ['on-device-practice-01-pruning-cnn', 'notebooks/on_device_ai/01_pruning_cnn.html'],
  ['on-device-practice-02-quantization-cnn', 'notebooks/on_device_ai/02_quantization_cnn.html'],
  ['on-device-practice-03-knowledge-distillation', 'notebooks/on_device_ai/03_knowledge_distillation.html'],
  ['on-device-practice-04-pruning-llm', 'notebooks/on_device_ai/04_pruning_llm.html'],
  ['on-device-practice-05-quantization-llm', 'notebooks/on_device_ai/05_quantization_llm.html'],
  ['vision-practice-01-resnet18-cifar10', 'notebooks/vision/01_resnet18_cifar10.html'],
  ['vision-practice-02-vit-cifar10', 'notebooks/vision/02_vit_cifar10.html'],
  ['vision-practice-03-detr', 'notebooks/vision/03_detr.html'],
  ['vision-practice-04-unet', 'notebooks/vision/04_unet.html'],
  ['llm-practice-01-vector-space', 'notebooks/language/01_vector_space.html'],
  ['llm-practice-02-dataset', 'notebooks/language/02_dataset.html'],
  ['llm-practice-03-attention', 'notebooks/language/03_attention.html'],
  ['llm-practice-03-attention-visualization', 'notebooks/language/03_attention_visualization.html'],
  ['llm-practice-04-gpt', 'notebooks/language/04_gpt.html'],
  ['llm-practice-05-pretraining', 'notebooks/language/05_pretraining.html'],
  ['llm-practice-06-classification-finetuning', 'notebooks/language/06_classification_finetuning.html'],
  ['llm-practice-06-lora-classification', 'notebooks/language/06_lora_classification.html'],
  ['llm-practice-07-instruction-finetuning', 'notebooks/language/07_instruction_finetuning.html'],
  ['llm-practice-07-dpo', 'notebooks/language/07_dpo.html'],

  ['rag-day1-practice-01-llama-index', 'notebooks/rag/day1/01_llama_index.html'],
  ['rag-day1-practice-02-rag-app', 'notebooks/rag/day1/02_rag_app.html']
]);
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
function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const p = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(p) : [p];
  });
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
 assertCheck(/sidebar-toggle/.test(html), 'index.html includes sidebar toggle');
  assertCheck(/AI_STUDY_NOTES/.test(render), 'render.js consumes AI_STUDY_NOTES');
  assertCheck(/fetch\(note\.path/.test(render), 'render.js fetches Markdown files');
  assertCheck(/notebookHtml/.test(render), 'render.js uses explicit notebookHtml manifest field');
 assertCheck(/protectHashInMathText/.test(render), 'render.js protects hash characters inside math text');
assertCheck(/protectMarkdownMath/.test(render), 'render.js protects TeX spans before Markdown parsing');
 assertCheck(/sidebar-collapsed/.test(render), 'render.js toggles collapsed sidebar state');
 assertCheck(/<iframe/.test(render) && /sandbox="allow-scripts"/.test(render), 'render.js creates sandboxed notebook iframe with script-only sandbox');
 assertCheck(/allow-scripts/.test(render), 'render.js allows notebook MathJax scripts');
  assertCheck(!/allow-same-origin/.test(render), 'render.js does not allow notebook iframe same-origin access');
  assertCheck(/notebook-layout/.test(styles) && /grid-template-columns/.test(styles), 'styles define notebook split-view grid');
 assertCheck(/sidebar-collapsed/.test(styles), 'styles define collapsed sidebar layout');
 assertCheck(/\.mermaid svg/.test(styles) && /height:\s*auto/.test(styles), 'styles keep Mermaid diagrams readable without forced oversizing');
  assertCheck(/@media \(max-width/.test(styles), 'styles define responsive fallback');
  assertCheck(/font-size:\s*18\.5px/.test(styles), 'styles increase study body font size');
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
  assertCheck(notes.length >= 25, 'manifest includes the expanded study note set');
  return Array.isArray(notes) ? notes : [];
}

function validateNoDeniedPublicFiles(notes) {
  for (const denied of DENIED_PUBLIC_PATHS) {
    assertCheck(!fs.existsSync(path.join(ROOT, denied)), `denied public file absent: ${denied}`);
    assertCheck(!notes.some((note) => note.path === denied.replace(/^study_viewer\//, '')), `denied manifest path absent: ${denied}`);
  }
  const allPublicFiles = walk(APP_DIR);
  for (const file of allPublicFiles) {
    const rel = path.relative(ROOT, file).replace(/\\/g, '/');
    assertCheck(!/study_viewer\/.*\/on_device_ai\/(ch01_|01_cnn_pruning_deep_review)/.test(rel), `no legacy On-Device draft file anywhere in public artifact: ${rel}`);
    assertCheck(!/study_viewer\/.*(answer|colab).*\.ipynb$/i.test(rel), `no answer/colab notebook variant anywhere in public artifact: ${rel}`);
  }
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
    if (note.kind === 'notebook') validateNotebookEntry(note, label);
  }
  assertCheck(notes.some((note) => /on_device_ai/.test(note.path)), 'manifest includes On-Device AI notes');
  assertCheck(notes.some((note) => /language/.test(note.path)), 'manifest includes Language notes');
  assertCheck(notes.some((note) => /vision/.test(note.path)), 'manifest includes Vision notes');
  assertCheck(notes.some((note) => /rag\//.test(note.path)), 'manifest includes RAG notes');
  assertCheck(notes.filter((note) => note.kind === 'notebook').length === 21, 'manifest includes twenty-one notebook practice entries');
  for (const [id, htmlPath] of EXPECTED_NOTEBOOKS) {
    const note = notes.find((item) => item.id === id);
    assertCheck(Boolean(note), `expected notebook entry exists: ${id}`);
    if (note) assertCheck(note.notebookHtml === htmlPath, `${id} maps to expected notebook HTML`);
  }
  assertCheck(!notes.some((note) => /answer|colab/i.test(`${note.path} ${note.notebookHtml || ''} ${note.sourceIpynb || ''}`)), 'manifest excludes answer/colab variants');
  assertCheck(!notes.some((note) => /05_|06_|DDPM|Stable|Generative/i.test(`${note.path} ${note.title}`) && /vision/i.test(note.path)), 'manifest excludes Vision 05+ generative material');
}

function validateNotebookEntry(note, label) {
  assertCheck(note.kind === 'notebook', `${label} has kind notebook`);
  assertCheck(typeof note.notebookHtml === 'string' && note.notebookHtml.startsWith('notebooks/'), `${label} notebookHtml stays under notebooks/`);
  assertCheck(typeof note.sourceIpynb === 'string' && note.sourceIpynb.endsWith('.ipynb'), `${label} sourceIpynb records original notebook`);
  const sourceGuidePath = path.join(ROOT, note.path);
  const publicGuidePath = path.join(APP_DIR, note.path);
  assertCheck(fs.existsSync(sourceGuidePath), `${label} source guide exists`);
  if (fs.existsSync(sourceGuidePath) && fs.existsSync(publicGuidePath)) {
    assertCheck(
      fs.readFileSync(sourceGuidePath, 'utf8') === fs.readFileSync(publicGuidePath, 'utf8'),
      `${label} public guide mirror matches source guide`
    );
  }
  const htmlPath = path.join(APP_DIR, note.notebookHtml);
  assertCheck(fs.existsSync(htmlPath), `${label} notebook HTML exists`);
  if (!fs.existsSync(htmlPath)) return;
  const htmlText = fs.readFileSync(htmlPath, 'utf8');
  assertCheck(htmlText.trim().length > 0, `${label} notebook HTML non-empty`);
  const htmlWithoutAllowedMathJax = htmlText
    .replace(/<script>\s*window\.MathJax = \{[\s\S]*?\};\s*<\/script>/g, '')
    .replace(/<script defer src="https:\/\/cdn\.jsdelivr\.net\/npm\/mathjax@3\/es5\/tex-svg\.js"><\/script>/g, '');
  assertCheck(/window\.MathJax/.test(htmlText), `${label} notebook HTML loads MathJax for Markdown math`);
  assertCheck(!/<\s*script\b/i.test(htmlWithoutAllowedMathJax), `${label} notebook HTML has no non-MathJax script tags`);
  assertCheck(!/<\s*(iframe|object|embed|form)\b/i.test(htmlWithoutAllowedMathJax), `${label} notebook HTML has no active embedded tags`);
  assertCheck(!/\son[a-zA-Z]+\s*=/i.test(htmlWithoutAllowedMathJax), `${label} notebook HTML has no inline event handlers`);
  assertCheck(!/javascript\s*:/i.test(htmlWithoutAllowedMathJax), `${label} notebook HTML has no javascript URLs`);
  assertCheck(/nb-cell/.test(htmlText), `${label} notebook HTML contains notebook cells`);
  assertCheck(/nb-code/.test(htmlText), `${label} notebook HTML contains code cells`);
  assertCheck(/Code Cell \d{3}/.test(htmlText), `${label} notebook HTML labels code cells by cell number`);
  assertCheck(/nb-code-line/.test(htmlText) && /nb-lineno/.test(htmlText), `${label} notebook HTML renders code line numbers`);
  assertCheck(!/In \[/.test(htmlText), `${label} notebook HTML omits old Jupyter In prompt labels`);
        assertCheck(/language-python/.test(htmlText), `${label} notebook HTML marks Python code language`);
        assertCheck(/tok-k/.test(htmlText) && /tok-s/.test(htmlText), `${label} notebook HTML includes static syntax highlight spans`);
  assertCheck(/nb-markdown/.test(htmlText), `${label} notebook HTML contains markdown cells`);
}

function validateConverterFixture() {
  const fixtureDir = path.join(ROOT, '.omx', 'tmp', 'notebook-fixture');
  fs.mkdirSync(fixtureDir, { recursive: true });
  const png1x1 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';
  const fixture = {
    cells: [
      { cell_type: 'markdown', metadata: {}, source: ['# Fixture\n', '| A | B |\n', '|---|---|\n', '| $x$ | `code` |\n', '![tiny](data:image/png;base64,' + png1x1 + ')\n', '<img src="data:image/png;base64,' + png1x1 + '" onclick="bad()" alt="raw">\n'] },
      { cell_type: 'code', execution_count: 1, metadata: {}, source: ['print("hello")'], outputs: [{ output_type: 'stream', name: 'stdout', text: ['hello\n'] }] },
      { cell_type: 'code', execution_count: 2, metadata: {}, source: ['display(img)'], outputs: [{ output_type: 'display_data', data: { 'image/png': png1x1, 'text/plain': '<image>' }, metadata: {} }] },
      { cell_type: 'code', execution_count: 3, metadata: {}, source: ['display(html)'], outputs: [{ output_type: 'display_data', data: { 'text/html': '<div onclick="bad()">safe<script>alert(1)</script><iframe src="x"></iframe><a href="javascript:alert(1)">bad</a><table><tr><td>ok</td></tr></table></div>' }, metadata: {} }] }
    ], metadata: {}, nbformat: 4, nbformat_minor: 5
  };
  const ipynb = path.join(fixtureDir, 'fixture.ipynb');
  const out = path.join(fixtureDir, 'fixture.html');
  fs.writeFileSync(ipynb, JSON.stringify(fixture), 'utf8');
  const result = spawnSync('python3', ['scripts/convert_ipynb_static.py', ipynb, out, 'Fixture'], { cwd: ROOT, encoding: 'utf8' });
  assertCheck(result.status === 0, 'notebook converter handles synthetic fixture');
  if (result.status !== 0) console.error((result.stderr || result.stdout || '').trim());
  if (!fs.existsSync(out)) return;
  const text = fs.readFileSync(out, 'utf8');
  assertCheck(/<table>/.test(text), 'fixture preserves markdown table');
  assertCheck(/hello/.test(text), 'fixture preserves stream output');
  assertCheck(/data:image\/png;base64/.test(text), 'fixture preserves image output');
  assertCheck(/<img src="data:image\/png;base64/.test(text), 'fixture preserves markdown and raw image tags safely');
  assertCheck(/<table>/.test(text) && /<td>ok<\/td>/.test(text), 'fixture preserves safe HTML table output');
 const fixtureWithoutAllowedMathJax = text
   .replace(/<script>\s*window\.MathJax = \{[\s\S]*?\};\s*<\/script>/g, '')
   .replace(/<script defer src="https:\/\/cdn\.jsdelivr\.net\/npm\/mathjax@3\/es5\/tex-svg\.js"><\/script>/g, '');
 assertCheck(/window\.MathJax/.test(text), 'fixture includes allowed MathJax scripts');
 assertCheck(!/<\s*script\b/i.test(fixtureWithoutAllowedMathJax), 'fixture strips non-MathJax script tags');
 assertCheck(!/<\s*(iframe|object|embed|form)\b/i.test(fixtureWithoutAllowedMathJax), 'fixture strips active embedded tags');
 assertCheck(!/\son[a-zA-Z]+\s*=/i.test(fixtureWithoutAllowedMathJax), 'fixture strips inline event handlers');
 assertCheck(!/javascript\s*:/i.test(fixtureWithoutAllowedMathJax), 'fixture strips javascript URLs');
}

validateStaticFiles();
const notes = loadManifest();
validateNoDeniedPublicFiles(notes);
validateNotes(notes);
validateConverterFixture();

if (failures > 0) {
  console.error(`\nRESULT FAIL ${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nRESULT PASS static AISP study viewer');
