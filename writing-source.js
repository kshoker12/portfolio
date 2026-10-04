// Loads a write-up from public/writing/. On localhost a staged copy wins over production.
(function () {
    const STAGING = 'writing/staging/';
    const PROD = 'writing/prod/';
    const isLocal = ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname);

    const fetchMarkdown = (url) =>
        fetch(url).then((res) => {
            const type = res.headers.get('content-type') || '';
            // The dev server can answer a missing file with index.html and a 200.
            if (!res.ok || type.includes('text/html')) throw new Error(`No write-up at ${url}`);
            return res.text();
        });

    const showStagingBadge = () => {
        const badge = document.createElement('div');
        badge.textContent = 'Staging preview';
        badge.style.cssText =
            'position:fixed;top:12px;right:12px;z-index:9999;padding:4px 10px;border-radius:6px;' +
            'background:#b45309;color:#fff;font:600 12px/1.6 system-ui,sans-serif;';
        document.body.appendChild(badge);
    };

    window.loadWriting = (slug, resolve = (path) => path) => {
        const prod = () =>
            fetchMarkdown(resolve(`${PROD}${slug}.md`)).then((markdown) => ({ markdown, assetBase: '' }));
        if (!isLocal) return prod();
        return fetchMarkdown(resolve(`${STAGING}${slug}.md`))
            .then((markdown) => {
                showStagingBadge();
                return { markdown, assetBase: STAGING };
            })
            .catch(prod);
    };

    window.rebaseWritingAssets = (root, assetBase) => {
        if (!assetBase) return;
        root.querySelectorAll('img[src], source[src], video[src]').forEach((el) => {
            const src = el.getAttribute('src');
            if (!src || /^(?:[a-z][a-z0-9+.-]*:|\/|#)/i.test(src)) return;
            el.setAttribute('src', assetBase + src);
        });
    };
})();
