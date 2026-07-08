#!/usr/bin/env node
'use strict';

function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_MODULE_PATH,
    'playwright',
    '/tmp/aisp-playwright/node_modules/playwright',
  ].filter(Boolean);
  const errors = [];
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch (error) {
      errors.push(`${candidate}: ${error.message}`);
    }
  }
  throw new Error(`Unable to load Playwright. Tried:\n${errors.join('\n')}`);
}

(async () => {
  const { chromium } = loadPlaywright();
  const baseUrl = process.env.AISP_SMOKE_URL || 'http://127.0.0.1:4173/';
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.goto(`${baseUrl}#on-device-practice-01-pruning-cnn`, { waitUntil: 'networkidle' });

    const layout = page.locator('.notebook-layout');
    const guide = page.locator('.notebook-guide');
    const iframe = page.locator('iframe.notebook-iframe');
    await layout.waitFor({ state: 'visible', timeout: 15000 });
    await guide.waitFor({ state: 'visible', timeout: 15000 });
    await iframe.waitFor({ state: 'visible', timeout: 15000 });

    const guideBox = await guide.boundingBox();
    const iframeBox = await iframe.boundingBox();
    const containerBox = await layout.boundingBox();
    if (!guideBox || !iframeBox || !containerBox) throw new Error('missing layout bounding box');

    const horizontallySeparated = guideBox.x + guideBox.width <= iframeBox.x + 16 || iframeBox.x + iframeBox.width <= guideBox.x + 16;
    const sameVerticalBand = Math.abs(guideBox.y - iframeBox.y) < 160;
    const substantialWidths = guideBox.width >= containerBox.width * 0.30 && iframeBox.width >= containerBox.width * 0.30;
    const noViewportOverflow = iframeBox.x + iframeBox.width <= 1440;
    if (!horizontallySeparated || !sameVerticalBand || !substantialWidths || !noViewportOverflow) {
      throw new Error(`not desktop side-by-side: guide=${JSON.stringify(guideBox)} iframe=${JSON.stringify(iframeBox)} container=${JSON.stringify(containerBox)}`);
    }

    const sandbox = await iframe.getAttribute('sandbox');
    if (sandbox === null) throw new Error('iframe sandbox attribute missing');
    if (sandbox !== 'allow-scripts') throw new Error(`unexpected sandbox attribute: ${sandbox}`);

    const frame = page.frame({ url: /01_pruning_cnn\.html/ });
    if (!frame) throw new Error('notebook frame not found by URL');
    await frame.locator('.nb-cell').first().waitFor({ state: 'visible', timeout: 15000 });
    await frame.locator('.nb-code').first().waitFor({ state: 'visible', timeout: 15000 });
    await frame.locator('mjx-container').first().waitFor({ state: 'visible', timeout: 20000 });
    const text = await frame.locator('body').innerText({ timeout: 15000 });
    if (!/Pruning for CNN|Assignment 1/i.test(text)) throw new Error('notebook body text missing expected title');

    console.log('RESULT PASS browser split-view smoke');
    console.log(JSON.stringify({ guideBox, iframeBox, containerBox, sandbox }));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
