// Drives the running Streamlit app and saves screenshots of every view.
// usage: node tests/ui_capture.js <outdir> <scenario label> [width height] [full|view]
const { chromium } = require('playwright');
(async () => {
  const out = process.argv[2] || 'shots';
  const scenario = process.argv[3] || 'Defective part';
  const W = parseInt(process.argv[4] || '1600'), H = parseInt(process.argv[5] || '1000');
  const mode = process.argv[6] || 'full';
  const fs = require('fs'); fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  await page.goto('http://localhost:8501', { waitUntil: 'networkidle' });
  await page.waitForSelector('text=AMSI · Adaptive Multi Scale Inspection', { timeout: 60000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/00_landing.png`, fullPage: mode === 'full' });
  await page.getByText(scenario, { exact: true }).click();
  await page.waitForTimeout(800);
  await page.getByRole('button', { name: '2 · Run inspection' }).click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${out}/01_running.png`, fullPage: false });
  await page.waitForSelector('text=Inspection complete', { timeout: 180000 });
  await page.waitForTimeout(4000);
  const tabs = ['1 · Camera survey (micron)', '2 · Scale switch', '3 · Confocal scans (sub micron)', '4 · Final report', 'How it works'];
  for (let i = 0; i < tabs.length; i++) {
    await page.getByRole('radio', { name: tabs[i], exact: true }).click();
    await page.waitForTimeout(3500);
    await page.screenshot({ path: `${out}/${String(i + 2).padStart(2, '0')}_tab${i + 1}.png`, fullPage: mode === 'full' });
  }
  // every confocal job
  await page.getByRole('radio', { name: tabs[2], exact: true }).click();
  await page.waitForTimeout(2000);
  const sel = page.getByRole('combobox').first();
  await sel.click(); await page.waitForTimeout(800);
  const opts = await page.getByRole('option').allInnerTexts();
  await page.keyboard.press('Escape');
  for (let k = 0; k < opts.length; k++) {
    await sel.click(); await page.waitForTimeout(600);
    await page.getByRole('option').nth(k).click();
    await page.waitForTimeout(3500);
    await page.screenshot({ path: `${out}/job_${k}.png`, fullPage: mode === 'full' });
  }
  fs.writeFileSync(`${out}/jobs.json`, JSON.stringify(opts));
  await browser.close();
})();
