document.addEventListener('DOMContentLoaded', function () {
    const referenceInput = document.getElementById('reference-input');
    const generatedInput = document.getElementById('generated-input');
    const rougeScoreDisplay = document.getElementById('rouge-score');
    const exactMatchScoreDisplay = document.getElementById('exact-match-score');
    const visualizationOutput = document.getElementById('rouge-visualization-output');

    /**
     * Compute the Longest Common Subsequence (LCS) length and the LCS itself
     * using dynamic programming.
     *
     * @param {Array} seq1 - First sequence (array of tokens)
     * @param {Array} seq2 - Second sequence (array of tokens)
     * @returns {Object} - { length: number, lcs: Array }
     */
    function computeLCS(seq1, seq2) {
        const m = seq1.length;
        const n = seq2.length;

        // Create DP table
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

        // Fill DP table
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (seq1[i - 1] === seq2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        // Backtrack to find the actual LCS
        const lcs = [];
        let i = m, j = n;
        while (i > 0 && j > 0) {
            if (seq1[i - 1] === seq2[j - 1]) {
                lcs.unshift(seq1[i - 1]);
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }

        return {
            length: dp[m][n],
            lcs: lcs
        };
    }

    /**
     * Calculate ROUGE-L score.
     * ROUGE-L = LCS_length / reference_length
     * (Simplified version - the full metric uses F1 combining precision and recall)
     *
     * @param {Array} reference - Reference sequence tokens
     * @param {Array} generated - Generated sequence tokens
     * @returns {number} - ROUGE-L score between 0 and 1
     */
    function calculateRougeL(reference, generated) {
        if (reference.length === 0 || generated.length === 0) {
            return 0.0;
        }

        const lcsResult = computeLCS(reference, generated);
        const lcsLength = lcsResult.length;

        // ROUGE-L as F1-score
        const recall = lcsLength / reference.length;
        const precision = lcsLength / generated.length;

        if (precision + recall === 0) {
            return 0.0;
        }

        const f1 = (2 * precision * recall) / (precision + recall);
        return f1;
    }

    /**
     * Calculate Exact Match score.
     * Returns 1.0 if sequences are identical, 0.0 otherwise.
     *
     * @param {Array} reference - Reference sequence tokens
     * @param {Array} generated - Generated sequence tokens
     * @returns {number} - Exact match score (0 or 1)
     */
    function calculateExactMatch(reference, generated) {
        if (reference.length !== generated.length) {
            return 0.0;
        }

        for (let i = 0; i < reference.length; i++) {
            if (reference[i] !== generated[i]) {
                return 0.0;
            }
        }

        return 1.0;
    }

    /**
     * Update the visualization and metrics display.
     */
    function updateVisualization() {
        const referenceText = referenceInput.value.trim();
        const generatedText = generatedInput.value.trim();

        if (!referenceText || !generatedText) {
            visualizationOutput.innerHTML = '<em>Please enter both sequences...</em>';
            rougeScoreDisplay.textContent = '0.00';
            exactMatchScoreDisplay.textContent = '0.00';
            return;
        }

        // Tokenize by whitespace
        const referenceTokens = referenceText.split(/\s+/);
        const generatedTokens = generatedText.split(/\s+/);

        // Calculate metrics
        const rougeL = calculateRougeL(referenceTokens, generatedTokens);
        const exactMatch = calculateExactMatch(referenceTokens, generatedTokens);

        // Update metric displays
        rougeScoreDisplay.textContent = rougeL.toFixed(2);
        exactMatchScoreDisplay.textContent = exactMatch.toFixed(2);

        // Compute LCS for visualization
        const lcsResult = computeLCS(referenceTokens, generatedTokens);
        const lcsTokens = lcsResult.lcs;

        // Create a Set for faster lookup
        const lcsSet = new Set(lcsTokens);

        // Track which LCS tokens we've already matched to avoid duplicates
        let lcsIndex = 0;

        // Visualize reference with LCS highlighted
        visualizationOutput.innerHTML = '';

        // Reference sequence
        const refDiv = document.createElement('div');
        refDiv.style.marginBottom = '10px';
        const refLabel = document.createElement('strong');
        refLabel.textContent = 'Reference: ';
        refDiv.appendChild(refLabel);

        for (let i = 0; i < referenceTokens.length; i++) {
            const token = referenceTokens[i];
            const span = document.createElement('span');
            span.textContent = token + ' ';

            // Check if this token is in the LCS at the current position
            if (lcsIndex < lcsTokens.length && token === lcsTokens[lcsIndex]) {
                span.classList.add('lcs-token');
                span.title = 'Part of LCS';
                lcsIndex++;
            }

            refDiv.appendChild(span);
        }

        visualizationOutput.appendChild(refDiv);

        // Reset LCS index for generated sequence
        lcsIndex = 0;

        // Generated sequence
        const genDiv = document.createElement('div');
        const genLabel = document.createElement('strong');
        genLabel.textContent = 'Generated: ';
        genDiv.appendChild(genLabel);

        for (let i = 0; i < generatedTokens.length; i++) {
            const token = generatedTokens[i];
            const span = document.createElement('span');
            span.textContent = token + ' ';

            // Check if this token is in the LCS at the current position
            if (lcsIndex < lcsTokens.length && token === lcsTokens[lcsIndex]) {
                span.classList.add('lcs-token');
                span.title = 'Part of LCS';
                lcsIndex++;
            }

            genDiv.appendChild(span);
        }

        visualizationOutput.appendChild(genDiv);
    }

    // Add event listeners
    referenceInput.addEventListener('input', updateVisualization);
    generatedInput.addEventListener('input', updateVisualization);

    // Initial update
    updateVisualization();
});
