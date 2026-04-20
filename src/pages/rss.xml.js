import rss from "@astrojs/rss";
import { getPublishedPosts } from "../lib/posts";
import { SITE_DESCRIPTION, SITE_TITLE, withBase } from "../lib/site";

export async function GET(context) {
  const posts = await getPublishedPosts();

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description ?? "",
      pubDate: post.data.publishDate,
      link: withBase(`/posts/${post.id}/`),
    })),
  });
}
