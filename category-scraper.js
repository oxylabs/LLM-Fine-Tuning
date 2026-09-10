// Use in Chrome > Dev Tools > console
// Must be on the amazon.com/gp/bestsellers page

(() => {
  const MAX_DEPTH = 2;
  const NAV_XPATH = '//ul[contains(@class, "nav-tree-all_style_zg-browse-group")]//li//a';
  const CATEGORY_ID = /\/zgbs\/([^\/?]+)(?:\/(\d+))?/;
  const NAV_LEVEL = /ref=zg_bs_nav_[^_]+_(\d+)/;

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function getLinks(doc, level) {
    const found = doc.evaluate(NAV_XPATH, doc, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
    const links = [], seen = new Set();
    for (let i = 0; i < found.snapshotLength; i++) {
      const a = found.snapshotItem(i);
      const href = a.getAttribute('href') || '';
      const [, slug, numericId] = href.match(CATEGORY_ID) || [];
      const [, navLevel] = href.match(NAV_LEVEL) || [];
      if (!slug) continue;
      if (navLevel !== undefined ? +navLevel !== level : level > 0 && !numericId) continue;
      const id = numericId || slug;
      if (seen.has(id)) continue;
      seen.add(id);
      links.push({ id, name: a.textContent.trim(), url: new URL(href, location.origin).href });
    }
    return links;
  }

  async function crawl(link, depth) {
    const node = { id: link.id, name: link.name, children: [] };
    if (depth >= MAX_DEPTH) return node;

    console.log(link.name);
    await sleep(500 + Math.random() * 200);
    let doc;
    try {
      const html = await (await fetch(link.url)).text();
      doc = new DOMParser().parseFromString(html, 'text/html');
    } catch (err) {
      console.warn('failed:', link.name, err.message);
      return node;
    }

    const children = getLinks(doc, depth).filter(c => c.id !== link.id);
    for (const child of children) {
      node.children.push(await crawl(child, depth + 1));
    }
    return node;
  }

  (async () => {
    const roots = getLinks(document, 0);
    console.log(`${roots.length} root categories, MAX_DEPTH = ${MAX_DEPTH}`);

    const tree = window.__categories = [];
    for (const link of roots) tree.push(await crawl(link, 1));

    console.log('done:', tree);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(tree, null, 2)], { type: 'application/json' }));
    a.download = 'categories.json';
    a.click();
  })();
})();
