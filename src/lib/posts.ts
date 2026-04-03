import { getCollection, type CollectionEntry } from "astro:content";

export type Post = CollectionEntry<"posts">;

export function slugifyTag(tag: string) {
  return tag
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export async function getPublishedPosts() {
  const posts = await getCollection("posts", ({ data }) => {
    if (import.meta.env.PROD) {
      return !data.draft;
    }

    return true;
  });

  return posts.sort(
    (left, right) =>
      right.data.publishDate.getTime() - left.data.publishDate.getTime(),
  );
}

export function getTagSummary(posts: Post[]) {
  const tagMap = new Map<string, number>();

  for (const post of posts) {
    for (const tag of post.data.tags) {
      tagMap.set(tag, (tagMap.get(tag) ?? 0) + 1);
    }
  }

  return [...tagMap.entries()]
    .map(([tag, count]) => ({ tag, count, slug: slugifyTag(tag) }))
    .sort((left, right) => {
      if (right.count !== left.count) {
        return right.count - left.count;
      }

      return left.tag.localeCompare(right.tag);
    });
}
