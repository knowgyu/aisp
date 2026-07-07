(function () {
  const notes = window.AI_STUDY_NOTES || [];
  const nav = document.querySelector('#note-nav');
  const panel = document.querySelector('#note-content');
  const title = document.querySelector('#note-title');
  const section = document.querySelector('#note-section');
  const rawLink = document.querySelector('#raw-link');
  const search = document.querySelector('#note-search');
  const validIds = new Set(notes.map((note) => note.id));

  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      securityLevel: 'strict',
      flowchart: { htmlLabels: false, curve: 'basis' },
      themeVariables: {
        background: '#101820',
        primaryColor: '#172331',
        primaryTextColor: '#eef7fb',
        primaryBorderColor: '#74b8ff',
        lineColor: '#98a9b3',
        secondaryColor: '#1d2c3a',
        tertiaryColor: '#0f171f'
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
      ? notes.filter((note) => `${note.section} ${note.title} ${note.path}`.toLowerCase().includes(query))
      : notes;
    return filtered.reduce((groups, note) => {
      if (!groups.has(note.section)) groups.set(note.section, []);
      groups.get(note.section).push(note);
      return groups;
    }, new Map());
  }

  function renderNav(activeId) {
    if (!nav) return;
    const groups = groupNotes(search && search.value);
    nav.innerHTML = Array.from(groups.entries()).map(([group, items]) => `
      <section class="nav-group">
        <h3>${escapeHtml(group)}</h3>
        ${items.map((note) => `
          <a href="#${encodeURIComponent(note.id)}" class="${note.id === activeId ? 'active' : ''}" data-note-id="${escapeHtml(note.id)}" aria-current="${note.id === activeId ? 'page' : 'false'}">
            <span>${escapeHtml(note.title)}</span>
            <small>${escapeHtml(note.path.replace(/^study_notes\//, ''))}</small>
          </a>
        `).join('')}
      </section>
    `).join('') || '<p class="empty-nav">검색 결과가 없습니다.</p>';
  }

  function dirname(path) {
    const index = path.lastIndexOf('/');
    return index >= 0 ? path.slice(0, index + 1) : '';
  }

  function isExternal(url) {
    return /^(https?:|mailto:|tel:|data:|#)/i.test(url || '');
  }

  function normalizeRelativeUrl(url, baseDir) {
    if (!url || isExternal(url)) return url;
    if (url.startsWith('/')) return url;
    const [path, hash] = url.split('#');
    const [cleanPath, query] = path.split('?');
    const stack = baseDir.split('/').filter(Boolean);
    for (const part of cleanPath.split('/')) {
      if (!part || part === '.') continue;
      if (part === '..') stack.pop();
      else stack.push(part);
    }
    const normalized = stack.join('/');
    return `${normalized}${query ? `?${query}` : ''}${hash ? `#${hash}` : ''}`;
  }

  function rewriteRelativeMarkdown(markdown, notePath) {
    const baseDir = dirname(notePath);
    const replaceUrl = (full, prefix, url, suffix) => {
      const trimmed = url.trim();
      if (!trimmed || trimmed.startsWith('<')) return full;
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
    if (!window.marked) {
      return `<pre>${escapeHtml(markdown)}</pre>`;
    }
    window.marked.setOptions({
      gfm: true,
      breaks: false,
      mangle: false,
      headerIds: true
    });
    const raw = window.marked.parse(markdown);
    return window.DOMPurify ? window.DOMPurify.sanitize(raw, { ADD_ATTR: ['target'] }) : raw;
  }

  function enhanceRenderedContent(note) {
    panel.querySelectorAll('a[href]').forEach((link) => {
      const href = link.getAttribute('href');
      if (/^https?:/i.test(href)) {
        link.target = '_blank';
        link.rel = 'noreferrer';
      }
    });

    panel.querySelectorAll('pre code.language-mermaid').forEach((code, index) => {
      const wrapper = document.createElement('div');
      wrapper.className = 'mermaid';
      wrapper.id = `mermaid-${note.id}-${index}`;
      wrapper.textContent = code.textContent;
      code.closest('pre').replaceWith(wrapper);
    });

    panel.querySelectorAll('img').forEach((img) => {
      img.loading = 'lazy';
      img.decoding = 'async';
    });

    if (window.mermaid) {
      window.mermaid.run({ nodes: panel.querySelectorAll('.mermaid') }).catch((error) => {
        console.warn('Mermaid render failed', error);
      });
    }

    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([panel]).catch((error) => {
        console.warn('MathJax render failed', error);
      });
    }
  }

  async function renderNote(id, pushHash) {
    const note = noteById(id);
    if (!note) return;
    renderNav(note.id);
    if (pushHash && window.location.hash !== `#${note.id}`) {
      history.pushState(null, '', `#${note.id}`);
    }
    title.textContent = note.title;
    section.textContent = note.section;
    rawLink.href = note.path;
    panel.innerHTML = '<p class="loading">노트를 불러오는 중입니다.</p>';

    try {
      const response = await fetch(note.path, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const markdown = rewriteRelativeMarkdown(await response.text(), note.path);
      panel.innerHTML = markdownToHtml(markdown);
      enhanceRenderedContent(note);
      document.title = `${note.title} · AI Study Notes`;
    } catch (error) {
      panel.innerHTML = `
        <section class="error-card">
          <h1>노트를 불러오지 못했습니다.</h1>
          <p>${escapeHtml(error.message)}</p>
          <p>로컬에서 볼 때는 <code>python3 -m http.server</code>로 정적 서버를 띄워야 fetch가 정상 동작합니다.</p>
        </section>
      `;
    }
  }

  if (nav) {
    nav.addEventListener('click', (event) => {
      const link = event.target.closest('a[data-note-id]');
      if (!link) return;
      event.preventDefault();
      renderNote(link.dataset.noteId, true);
    });
  }

  if (search) {
    search.addEventListener('input', () => renderNav(normalizeHash()));
  }

  window.addEventListener('hashchange', () => renderNote(normalizeHash(), false));
  renderNote(normalizeHash(), false);
})();
