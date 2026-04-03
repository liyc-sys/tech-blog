export const SITE_TITLE = "Build Notes";
export const SITE_DESCRIPTION =
  "A focused engineering blog for technical writeups, debugging notes, and system design lessons.";
export const SITE_NAV = [
  { href: "/", label: "Latest" },
  { href: "/posts/", label: "Posts" },
  { href: "/tags/", label: "Tags" },
  { href: "/rss.xml", label: "RSS" },
];

export function withBase(path: string) {
  const base = import.meta.env.BASE_URL;
  const normalizedPath = path.startsWith("/") ? path.slice(1) : path;

  return base === "/" ? `/${normalizedPath}` : `${base}${normalizedPath}`;
}
