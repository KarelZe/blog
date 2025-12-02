document.addEventListener('DOMContentLoaded', function () {
    const slider = document.getElementById('k-slider');
    const kValueDisplay = document.getElementById('k-value');
    const textContainer = document.getElementById('goldfish-text');
    const maskTypeRadios = document.getElementsByName('mask-type');
    const inputText = document.getElementById('input-text');

    // Simple hash function for demonstration
    // Hashes a string to an integer
    function simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }

    function updateMask() {
        const k = parseInt(slider.value);
        kValueDisplay.textContent = k;

        // Get selected mask type
        let maskType = 'static';
        for (const radio of maskTypeRadios) {
            if (radio.checked) {
                maskType = radio.value;
                break;
            }
        }

        // Get text from input
        const sampleText = inputText.value.trim();
        if (!sampleText) {
            textContainer.innerHTML = '<em>Please enter some text...</em>';
            return;
        }

        const tokens = sampleText.split(' ');

        // Clear current text
        textContainer.innerHTML = '';

        tokens.forEach((token, index) => {
            const span = document.createElement('span');
            span.textContent = token + ' ';

            let isMasked = false;

            if (k === 1) {
                isMasked = true;
            } else if (maskType === 'static') {
                // Static mask: drop every k-th token
                // We use (index + 1) % k === 0 to make it 1-based for intuition (e.g. k=4 masks 4th, 8th...)
                isMasked = ((index + 1) % k) === 0;
            } else {
                // Hashed mask: depends on context
                // For demo, we use the previous token as context (h=1)
                // In reality, h is larger (e.g., 13)
                const context = index > 0 ? tokens[index - 1] : "START";
                const hash = simpleHash(context + token); // Hash context + current token (or just context)
                // The paper says: hash(context) < 1/k.
                // Equivalent to: hash(context) % k === 0
                isMasked = (hash % k) === 0;
            }

            if (isMasked) {
                span.classList.add('masked-token');
                span.title = "Masked (Loss ignored)";
            }

            textContainer.appendChild(span);
        });
    }

    slider.addEventListener('input', updateMask);
    for (const radio of maskTypeRadios) {
        radio.addEventListener('change', updateMask);
    }
    inputText.addEventListener('input', updateMask);

    // Initial call
    updateMask();
});
