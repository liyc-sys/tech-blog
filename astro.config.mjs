import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const owner = process.env.GITHUB_REPOSITORY_OWNER ?? "liyc-sys";
const repo = process.env.GITHUB_REPOSITORY?.split("/")[1] ?? "tech-blog";
const isGitHubPagesBuild = process.env.GITHUB_ACTIONS === "true";
const site = process.env.SITE_URL ?? `https://${owner}.github.io`;
const base =
  isGitHubPagesBuild && repo !== `${owner}.github.io` ? `/${repo}` : "/";

export default defineConfig({
  site,
  base,
  integrations: [mdx(), sitemap()],
  trailingSlash: "always",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: "github-dark",
      wrap: true,
    },
  },
});
