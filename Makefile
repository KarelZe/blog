# Makefile for Hugo blog with Pandoc citations

.PHONY: dev build process-citations clean

# Local development server (without citation processing)
dev:
	hugo server --buildDrafts

# Full build with citation processing
build: process-citations
	@echo "Checking for unreferenced files in page bundles..."
	@find content/post -mindepth 1 -maxdepth 1 -type d | while read dir; do \
		[ -f "$$dir/index.md" ] || [ -f "$$dir/index.md.draft" ] && ./scripts/clean-pagebundle.sh "$$dir"; \
	done || true
	hugo --gc --minify

# Process citations in all content
process-citations:
	./scripts/process-citations.sh

# Clean generated files
clean:
	rm -rf public resources
