(function () {
  const notes = window.AI_STUDY_NOTES || [];
  const nav = document.querySelector('#note-nav');
  const panel = document.querySelector('#note-content');
  const title = document.querySelector('#note-title');
  const section = document.querySelector('#note-section');
  const rawLink = document.querySelector('#raw-link');
  const search = document.querySelector('#note-search');
  const sideNote = document.querySelector('.side-note p');
  const sidebarToggle = document.querySelector('#sidebar-toggle');
  const validIds = new Set(notes.map((note) => note.id));

  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      securityLevel: 'strict',
      flowchart: { htmlLabels: false, curve: 'basis', useMaxWidth: false },
      sequence: { useMaxWidth: false },
      themeVariables: {
        background: '#101820',
        primaryColor: '#172331',
        primaryTextColor: '#eef7fb',
        primaryBorderColor: '#74b8ff',
        lineColor: '#98a9b3',
        secondaryColor: '#1d2c3a',
        tertiaryColor: '#0f171f',
        fontSize: '18px'
      }
    });
  }

  function escapeHtml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function normalizeHash() {
    const raw = decodeURIComponent(window.location.hash.replace(/^#/, ''));
    if (validIds.has(raw)) return raw;
    const pair = raw.match(/^note=(.+)$/);
    if (pair && validIds.has(pair[1])) return pair[1];
    return notes[0] ? notes[0].id : '';
  }

  function noteById(id) {
    return notes.find((note) => note.id === id) || notes[0];
  }

  function groupNotes(filterText) {
    const query = String(filterText || '').trim().toLowerCase();
    const filtered = query
      ? notes.filter((note) => `${note.section} ${note.title} ${note.path} ${note.sourceIpynb || ''}`.toLowerCase().includes(query))
      : notes;
    return filtered.reduce((groups, note) => {
      if (!groups.has(note.section)) groups.set(note.section, []);
      groups.get(note.section).push(note);
      return groups;
    }, new Map());
  }

  function renderNav(activeId) {
    const groups = groupNotes(search.value);
    nav.innerHTML = Array.from(groups.entries()).map(([group, items]) => `
      <section class="nav-group">
        <h3>${escapeHtml(group)}</h3>
        ${items.map((note) => `
          <a href="#${encodeURIComponent(note.id)}" class="${note.id === activeId ? 'active' : ''} ${note.kind === 'notebook' ? 'notebook-link' : ''}" data-note-id="${escapeHtml(note.id)}" aria-current="${note.id === activeId ? 'page' : 'false'}">
            <span>${escapeHtml(note.title)}</span>
            <small>${note.kind === 'notebook' ? 'Notebook + Guide' : escapeHtml(note.path.replace(/^study_notes\//, ''))}</small>
          </a>
        `).join('')}
      </section>
    `).join('') || '<p class="empty-nav">검색 결과가 없습니다.</p>';
  }

  function baseDirOf(path) {
    const index = path.lastIndexOf('/');
    return index >= 0 ? path.slice(0, index + 1) : '';
  }

  function isExternal(url) {
    return /^(https?:|mailto:|tel:|data:|#)/i.test(url) || url.startsWith('/');
  }

  function normalizeRelativeUrl(url, baseDir) {
    const [cleanPath, suffix = ''] = url.split(/(?=[?#])/);
    const stack = baseDir.split('/').filter(Boolean);
    for (const part of cleanPath.split('/')) {
      if (!part || part === '.') continue;
      if (part === '..') stack.pop();
      else stack.push(part);
    }
    return `${stack.join('/')}${suffix}`;
  }

  function rewriteRelativeUrls(markdown, notePath) {
    const baseDir = baseDirOf(notePath);
    const replaceUrl = (full, prefix, url, suffix) => {
      const trimmed = url.trim();
      if (!trimmed || trimmed.startsWith('<') || isExternal(trimmed)) return full;
      return `${prefix}${normalizeRelativeUrl(trimmed, baseDir)}${suffix}`;
    };
    return markdown
      .replace(/(!\[[^\]]*\]\()([^\s)]+)(\))/g, replaceUrl)
      .replace(/(?<!!)\[([^\]]+)\]\(([^\s)]+)(\))/g, (full, label, url, suffix) => {
        if (isExternal(url) || url.endsWith('.md')) return full;
        return `[${label}](${normalizeRelativeUrl(url, baseDir)}${suffix}`;
      });
  }

  function markdownToHtml(markdown) {
    if (!window.marked) return `<pre>${escapeHtml(markdown)}</pre>`;
    const protectedMath = protectMarkdownMath(markdown);
    window.marked.setOptions({ gfm: true, breaks: false, mangle: false, headerIds: true });
    const raw = window.marked.parse(protectedMath.markdown);
    const safe = window.DOMPurify ? window.DOMPurify.sanitize(raw, { ADD_ATTR: ['target'] }) : raw;
    return protectedMath.restore(safe);
  }

  function isEscaped(text, index) {
    let slashCount = 0;
    for (let back = index - 1; back >= 0 && text[back] === '\\'; back -= 1) slashCount += 1;
    return slashCount % 2 === 1;
  }

  function findClosingDelimiter(text, start, delimiter) {
    for (let index = start; index < text.length; index += 1) {
      if (text.startsWith(delimiter, index) && !isEscaped(text, index)) return index;
    }
    return -1;
  }

  function copyUntilLineEnd(text, start) {
    const end = text.indexOf('\n', start);
    return end < 0 ? text.length : end + 1;
  }

  function protectMarkdownMath(markdown) {
    const tokens = [];
    const makeToken = (value) => {
      const token = `@@AISP_MATH_${tokens.length}@@`;
      tokens.push({ token, value });
      return token;
    };

    let output = '';
    let index = 0;
    let lineStart = true;
    let fence = '';
    while (index < markdown.length) {
      if (lineStart && (markdown.startsWith('```', index) || markdown.startsWith('~~~', index))) {
        const marker = markdown.slice(index, index + 3);
        if (!fence) fence = marker;
        else if (marker === fence) fence = '';
        const end = copyUntilLineEnd(markdown, index);
        output += markdown.slice(index, end);
        index = end;
        lineStart = true;
        continue;
      }

      if (fence) {
        output += markdown[index];
        lineStart = markdown[index] === '\n';
        index += 1;
        continue;
      }

      if (markdown[index] === '`') {
        const match = markdown.slice(index).match(/^`+/);
        const delimiter = match[0];
        const end = markdown.indexOf(delimiter, index + delimiter.length);
        if (end >= 0) {
          const next = end + delimiter.length;
          output += markdown.slice(index, next);
          lineStart = /\n$/.test(output);
          index = next;
          continue;
        }
      }

      const candidates = [
        ['$$', '$$'],
        ['\\[', '\\]'],
        ['\\(', '\\)']
      ];
      let protectedSegment = false;
      for (const [open, close] of candidates) {
        if (!markdown.startsWith(open, index)) continue;
        const start = index + open.length;
        const end = findClosingDelimiter(markdown, start, close);
        if (end < 0) continue;
        const next = end + close.length;
        output += makeToken(markdown.slice(index, next));
        index = next;
        lineStart = false;
        protectedSegment = true;
        break;
      }
      if (protectedSegment) continue;

      if (markdown[index] === '$' && markdown[index + 1] !== '$' && !isEscaped(markdown, index)) {
        const end = findClosingDelimiter(markdown, index + 1, '$');
        if (end >= 0) {
          output += makeToken(markdown.slice(index, end + 1));
          index = end + 1;
          lineStart = false;
          continue;
        }
      }

      output += markdown[index];
      lineStart = markdown[index] === '\n';
      index += 1;
    }

    return {
      markdown: output,
      restore(html) {
        return tokens.reduce((result, item) => result.replaceAll(item.token, escapeHtml(item.value)), html);
      }
    };
  }

  function escapeUnescapedHash(value) {
    let output = '';
    for (let index = 0; index < value.length; index += 1) {
      const char = value[index];
      if (char !== '#') {
        output += char;
        continue;
      }
      let slashCount = 0;
      for (let back = index - 1; back >= 0 && value[back] === '\\'; back -= 1) slashCount += 1;
      output += slashCount % 2 === 0 ? '\\#' : '#';
    }
    return output;
  }

  function findClosingMath(text, start, delimiter) {
    for (let index = start; index < text.length; index += 1) {
      if (!text.startsWith(delimiter, index)) continue;
      let slashCount = 0;
      for (let back = index - 1; back >= 0 && text[back] === '\\'; back -= 1) slashCount += 1;
      if (slashCount % 2 === 0) return index;
    }
    return -1;
  }

  function protectHashInMathText(text) {
    let output = '';
    let index = 0;
    while (index < text.length) {
      let open = '';
      let close = '';
      if (text.startsWith('$$', index)) {
        open = '$$';
        close = '$$';
      } else if (text.startsWith('\\(', index)) {
        open = '\\(';
        close = '\\)';
      } else if (text.startsWith('\\[', index)) {
        open = '\\[';
        close = '\\]';
      } else if (text[index] === '$' && text[index + 1] !== '$') {
        open = '$';
        close = '$';
      }
      if (!open) {
        output += text[index];
        index += 1;
        continue;
      }
      const contentStart = index + open.length;
      const contentEnd = findClosingMath(text, contentStart, close);
      if (contentEnd < 0) {
        output += text.slice(index);
        break;
      }
      output += open + escapeUnescapedHash(text.slice(contentStart, contentEnd)) + close;
      index = contentEnd + close.length;
    }
    return output;
  }

  function protectMathTextNodes(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || parent.closest('pre, code, .mermaid, mjx-container')) return NodeFilter.FILTER_REJECT;
        return /#/.test(node.nodeValue) && /(\$|\\\(|\\\[)/.test(node.nodeValue)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_SKIP;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      node.nodeValue = protectHashInMathText(node.nodeValue);
    });
  }

  function enhanceRenderedContent(root, note) {
    root.querySelectorAll('a[href]').forEach((link) => {
      const href = link.getAttribute('href');
      if (/^https?:/i.test(href)) {
        link.target = '_blank';
        link.rel = 'noreferrer';
      }
    });
    root.querySelectorAll('pre code.language-mermaid').forEach((code, index) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'mermaid';
      wrapper.id = `mermaid-${note.id}-${index}`;
      wrapper.textContent = code.textContent;
      code.closest('pre').replaceWith(wrapper);
    });
    root.querySelectorAll('img').forEach((img) => {
      img.loading = 'lazy';
      img.decoding = 'async';
    });
    protectMathTextNodes(root);
    if (window.mermaid) {
      window.mermaid.run({ nodes: root.querySelectorAll('.mermaid') }).catch((error) => console.warn('Mermaid render failed', error));
    }
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([root]).catch((error) => console.warn('MathJax render failed', error));
    }
  }

  async function loadMarkdown(note) {
    const response = await fetch(note.path, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return rewriteRelativeUrls(await response.text(), note.path);
  }

  function setChrome(note) {
    renderNav(note.id);
    if (window.location.hash !== `#${note.id}`) history.pushState(null, '', `#${note.id}`);
    title.textContent = note.title;
    section.textContent = note.section;
    rawLink.href = note.path;
    rawLink.textContent = note.kind === 'notebook' ? 'Guide MD' : 'Raw MD';
    if (sideNote) {
      sideNote.textContent = note.kind === 'notebook'
        ? '실습 페이지는 왼쪽 설명과 오른쪽 원본 노트북 HTML을 함께 보여줍니다. 노트북은 원본 셀과 실행 결과를 유지합니다.'
        : '왼쪽에서 노트를 고르면 오른쪽에 Markdown 원문이 문서형으로 렌더링됩니다. 수식, Mermaid, 이미지를 함께 확인합니다.';
    }
  }

  async function renderNote(id) {
    const note = noteById(id);
    if (!note) return;
    setChrome(note);
    panel.className = note.kind === 'notebook' ? 'note-content notebook-content' : 'note-content markdown-body';
    panel.innerHTML = '<p class="loading">노트를 불러오는 중입니다.</p>';
    try {
      const markdown = await loadMarkdown(note);
      const guideHtml = markdownToHtml(markdown);
      if (note.kind === 'notebook') {
        panel.innerHTML = `
          <div class="notebook-layout" data-note-kind="notebook">
            <section class="notebook-guide markdown-body" aria-label="Curated notebook guide">${guideHtml}</section>
              <aside class="notebook-frame-panel" aria-label="Original notebook HTML">
                <div class="notebook-frame-head">
                  <strong>원본 실습 Notebook</strong>
                  <span>
                  ${note.practiceIpynb ? `<a href="${escapeHtml(note.practiceIpynb)}" download>원본 .ipynb</a>` : ''}
                  <a href="${escapeHtml(note.notebookHtml)}" target="_blank" rel="noreferrer">HTML 새 탭</a>
                </span>
              </div>
              <iframe class="notebook-iframe" title="${escapeHtml(note.title)} 원본 실습 노트북 HTML" src="${escapeHtml(note.notebookHtml)}" sandbox="allow-scripts" loading="lazy"></iframe>
            </aside>
          </div>
        `;
      } else {
        panel.innerHTML = guideHtml;
      }
      enhanceRenderedContent(panel, note);
      document.title = `${note.title} · AI Study Notes`;
    } catch (error) {
      panel.className = 'note-content markdown-body';
      panel.innerHTML = `
        <section class="error-card">
          <h1>노트를 불러오지 못했습니다.</h1>
          <p>${escapeHtml(error.message)}</p>
          <p>로컬에서는 <code>python3 -m http.server --directory study_viewer</code>로 실행해야 fetch가 동작합니다.</p>
        </section>
      `;
    }
  }

  nav.addEventListener('click', (event) => {
    const link = event.target.closest('a[data-note-id]');
    if (!link) return;
    event.preventDefault();
    renderNote(link.dataset.noteId);
  });
  search.addEventListener('input', () => renderNav(normalizeHash()));
  if (sidebarToggle) {
    const setSidebarCollapsed = (collapsed) => {
      document.body.classList.toggle('sidebar-collapsed', collapsed);
      sidebarToggle.setAttribute('aria-expanded', String(!collapsed));
      sidebarToggle.textContent = collapsed ? '목차 열기' : '목차 접기';
    };
    setSidebarCollapsed(window.localStorage.getItem('aiStudySidebarCollapsed') === '1');
    sidebarToggle.addEventListener('click', () => {
      const collapsed = !document.body.classList.contains('sidebar-collapsed');
      window.localStorage.setItem('aiStudySidebarCollapsed', collapsed ? '1' : '0');
      setSidebarCollapsed(collapsed);
    });
  }
  window.addEventListener('hashchange', () => renderNote(normalizeHash()));
  renderNav(normalizeHash());
  renderNote(normalizeHash());
})();
