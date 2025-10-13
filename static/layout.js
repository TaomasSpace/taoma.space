(function () {
    function loadFragment(selector, url) {
        var placeholder = document.querySelector(selector);
        if (!placeholder) {
            return;
        }

        try {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url, false);
            xhr.send(null);

            if (xhr.status >= 200 && xhr.status < 300) {
                var template = document.createElement('template');
                template.innerHTML = xhr.responseText.trim();
                var fragment = template.content;
                placeholder.replaceWith(fragment);
            } else {
                console.error('Failed to load fragment', url, xhr.status);
            }
        } catch (err) {
            console.error('Failed to load fragment', url, err);
        }
    }

    loadFragment('[data-fragment="header"]', '/static/header.html');
    loadFragment('[data-fragment="footer"]', '/static/footer.html');
})();
