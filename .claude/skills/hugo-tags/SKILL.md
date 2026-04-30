---
name: hugo-tags
description: Use when improving, normalizing, auditing, or generating tags
  in Hugo blog post frontmatter. Triggers on requests to "fix tags",
  "retag posts", "clean up taxonomy", or when adding tags to new posts
  in a Hugo content/ directory. Do NOT use for Jekyll, Astro, or other
  SSGs.
---

# Hugo tag improvement

## When to use
- User points at a Hugo `content/` directory (or a single `.md` post) and
  wants tags reviewed, added, or normalized.
- User has just written a new post and wants tag suggestions consistent
  with their existing taxonomy.

## Workflow

1. **Inventory existing tags first.** Before suggesting anything, scan
   all posts under `content/` and build a frequency map of current tags.
   Surface this to the user. Never invent new tags before seeing what
   already exists.
   **IMPORTANT:** if index.md.draft files exists, edit these instead of index.md, and ignore the latter.

2. **Apply these conventions** (edit to match the user's actual rules):
   - kebab-case, lowercase, ASCII only
   - 3–6 tags per post, no more
   - Prefer reusing an existing tag over creating a near-duplicate
     (e.g. don't add `python3` if `python` exists)
   - Tags describe topic, not format ("essay", "notes" go in
     `categories`, not `tags`)

3. **Edit frontmatter in place.** Posts use YAML frontmatter delimited
   by `---`. Preserve key order, indentation style, and surrounding
   whitespace. The `tags` field may be either inline (`tags: [a, b, c]`)
   or block style:

       tags:
         - a
         - b

   Match whichever style the file already uses — don't convert between
   them.

4. **Show a diff before bulk changes.** For runs touching >3 posts,
   present the proposed changes and wait for confirmation.

## Edge cases
- Draft posts (`draft: true`): include unless user says otherwise.
- Posts with no tags at all: suggest based on title + first 200 words,
  drawing only from the existing tag vocabulary unless the user
  approves a new tag.
- `tags` vs `Tags`: match the casing already used in that file.
