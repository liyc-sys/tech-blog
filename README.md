# Tech Blog

A lean Astro blog set up for publishing technical articles to GitHub Pages.

## Stack

- Astro
- Markdown and MDX content collections
- GitHub Pages via GitHub Actions
- RSS and sitemap enabled

## Local development

```bash
npm install
npm run dev
```

## Writing flow

1. Add a new `.md` or `.mdx` file in `src/content/posts/`.
2. Fill in the frontmatter:

```md
---
title: "Post title"
description: "One-sentence summary"
publishDate: 2026-04-03
tags:
  - debugging
  - systems
draft: false
---
```

3. Push to `main`. GitHub Actions will build and deploy the site automatically.

## Deployment

The included workflow deploys to GitHub Pages. For a repository named
`tech-blog`, the site will publish to:

`https://liyc-sys.github.io/tech-blog/`

If you rename the repository later, the Astro config derives the correct base
path from GitHub Actions automatically.
