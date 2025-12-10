document.addEventListener('DOMContentLoaded', function () {
    const slider = document.getElementById('k-slider');
    const kValueDisplay = document.getElementById('k-value');
    const textContainer = document.getElementById('goldfish-text');
    const maskTypeRadios = document.getElementsByName('mask-type');
    const inputText = document.getElementById('input-text');

    // Simulated hash table (fixed random values for consistency)
    // In a real app, this would be a large array of random floats [0, 1)
    const TABLE_SIZE = 1000;
    const HASH_TABLE = new Float32Array(TABLE_SIZE);
    // Seeded random generator for reproducibility across reloads
    let seed = 12345;
    function random() {
        const x = Math.sin(seed++) * 10000;
        return x - Math.floor(x);
    }
    for (let i = 0; i < TABLE_SIZE; i++) {
        HASH_TABLE[i] = random();
    }

    // Simple string hash to map context to table index
    function getTableIndex(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash) % TABLE_SIZE;
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

        const tokens = sampleText.split(/\s+/); // Split by whitespace
        const contextWidth = 4; // h=4, matching the python example

        // Clear current text
        textContainer.innerHTML = '';

        tokens.forEach((token, index) => {
            const span = document.createElement('span');

            let isMasked = false;

            if (k === 1) {
                isMasked = true;
            } else if (maskType === 'static') {
                // Static mask: drop every k-th token
                // We use (index + 1) % k === 0 to make it 1-based for intuition (e.g. k=4 masks 4th, 8th...)
                isMasked = ((index + 1) % k) === 0;
            } else {
                // Hashed mask (hash-table strategy)

                // Cold Start: First (h-1) tokens are NEVER masked
                if (index < contextWidth - 1) {
                    isMasked = false;
                } else {
                    // Context window: [index - (h-1), ..., index]
                    // This window has length 'contextWidth' and ends at the current token
                    // We join them to simulate the "product of tokens"
                    const start = index - (contextWidth - 1);
                    const end = index + 1; // slice is exclusive
                    const windowTokens = tokens.slice(start, end);
                    const contextStr = windowTokens.join(" ");

                    const tableIndex = getTableIndex(contextStr);
                    const randomValue = HASH_TABLE[tableIndex];

                    // Drop if random value < 1/k
                    isMasked = randomValue < (1.0 / k);
                }
            }

            if (isMasked) {
                // adapted from paper appendix.
                span.textContent = '[DROP] ';
                span.classList.add('masked-token');
                span.title = "Masked (Loss ignored): " + token;
            } else {
                span.textContent = token + ' ';
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
