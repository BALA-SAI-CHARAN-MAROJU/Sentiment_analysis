"""
src/social_media_fetcher.py - Real-Time Social Media Data Collection Module
Fetches real-time social media posts using ONLY free public endpoints:
- Hacker News: Public Algolia search API (real community discussions & comments)
- Mastodon: Public timelines from decentralized Fediverse instances (HTML stripped)
- GitHub: Public open-source issue discussions & developer community comments
- Bluesky: Public search feed with automatic fallback
- Reddit: PRAW (if keys present) or Reddit Public JSON with fallback
- Demonstration Mode: Automatically loads 20 curated local sample posts if
  offline, rate-limited, or blocked.
"""

from datetime import datetime
import html
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import requests

from config import (
    MASTODON_ACCESS_TOKEN,
    MASTODON_INSTANCE,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    SAMPLE_LIVE_POSTS_FILE,
    YOUTUBE_API_KEY,
)
from src.utils import logger


class SocialMediaFetcher:
    """
    Unified manager for real-time social media data collection.
    Guarantees standard output format and never raises unhandled exceptions.
    """

    def __init__(self):
        self.reddit_client_id = REDDIT_CLIENT_ID
        self.reddit_client_secret = REDDIT_CLIENT_SECRET
        self.reddit_user_agent = REDDIT_USER_AGENT or "SocialSentimentAnalyzer/1.0"
        self.mastodon_instance = MASTODON_INSTANCE or "https://mastodon.social"
        self.mastodon_token = MASTODON_ACCESS_TOKEN
        self.youtube_api_key = YOUTUBE_API_KEY
        self.sample_file = SAMPLE_LIVE_POSTS_FILE

    def load_offline_samples(self, source_filter: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Load demonstration social media posts from local JSON file.
        Guarantees that the project demo always works even if completely offline.
        """
        if not self.sample_file.exists():
            logger.warning(f"Sample posts file not found at {self.sample_file}")
            return []

        try:
            with open(self.sample_file, "r", encoding="utf-8") as f:
                posts = json.load(f)

            if source_filter and source_filter.lower() != "all":
                filter_key = source_filter.lower()
                posts = [p for p in posts if filter_key in p.get("source", "").lower()]

            return posts[:limit]
        except Exception as e:
            logger.error(f"Failed to load offline sample posts: {e}")
            return []

    def fetch_hacker_news_posts(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch real-time tech discussions and comments from Hacker News via public Algolia API.
        100% free, no API key, zero sign-up.
        """
        q = query or ""
        url = "https://hn.algolia.com/api/v1/search_by_date"
        params = {
            "tags": "comment",
            "hitsPerPage": min(limit * 2, 40),
        }
        if q:
            params["query"] = q

        posts = []
        try:
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code == 200:
                hits = resp.json().get("hits", [])
                for h in hits:
                    raw_text = h.get("comment_text", "")
                    if not raw_text:
                        continue

                    # Strip HTML formatting from HN comments
                    clean_comment = re.sub(r"<[^>]+>", " ", raw_text)
                    clean_comment = html.unescape(clean_comment).strip()
                    # Filter very short or noisy comments
                    if len(clean_comment) < 20 or len(clean_comment) > 400:
                        continue

                    created_at = h.get("created_at", "")
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    story_title = h.get("story_title") or "Discussion"
                    item_id = h.get("objectID") or h.get("story_id") or ""
                    post_url = f"https://news.ycombinator.com/item?id={item_id}" if item_id else "https://news.ycombinator.com"

                    posts.append({
                        "text": clean_comment,
                        "source": f"Hacker News ({story_title[:30]}...)" if len(story_title) > 30 else f"Hacker News ({story_title})",
                        "timestamp": dt_str,
                        "created_at": dt_str,
                        "url": post_url,
                    })

                    if len(posts) >= limit:
                        break
        except Exception as e:
            logger.warning(f"Hacker News fetch error ({e}). Will check fallback.")

        if not posts:
            posts = self.load_offline_samples(source_filter="hacker news", limit=limit)
            if not posts:
                posts = self.load_offline_samples(limit=limit)

        return posts[:limit]

    def fetch_mastodon_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch public posts from Mastodon public timeline API.
        Strips HTML tags to extract clean text.
        """
        candidate_instances = [
            "https://fosstodon.org",
            "https://mastodon.online",
            self.mastodon_instance,
        ]

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SentimentProject/1.0"}
        if self.mastodon_token:
            headers["Authorization"] = f"Bearer {self.mastodon_token}"

        posts = []
        for instance in candidate_instances:
            if not instance:
                continue
            instance_clean = instance.rstrip("/")
            url = f"{instance_clean}/api/v1/timelines/public?limit={max(limit, 5)}"
            try:
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data:
                        content_html = item.get("content", "")
                        clean_content = re.sub(r"<[^>]+>", " ", content_html)
                        clean_content = html.unescape(clean_content).strip()

                        raw_date = item.get("created_at", "")
                        try:
                            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        post_url = item.get("url") or f"{instance_clean}"

                        if clean_content and 15 < len(clean_content) < 400:
                            posts.append({
                                "text": clean_content,
                                "source": f"Mastodon ({Path(instance_clean).name})",
                                "timestamp": dt_str,
                                "created_at": dt_str,
                                "url": post_url,
                            })
                        if len(posts) >= limit:
                            break

                    if posts:
                        break
            except Exception as e:
                logger.warning(f"Mastodon instance {instance_clean} error: {e}")

        if not posts:
            posts = self.load_offline_samples(source_filter="mastodon", limit=limit)

        return posts[:limit]

    def fetch_github_comments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch real developer feedback & issue comments from popular open source projects on GitHub.
        100% free public REST API, zero API key required.
        """
        repos = ["facebook/react", "python/cpython", "microsoft/vscode"]
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SentimentProject/1.0"}

        posts = []
        for repo in repos:
            url = f"https://api.github.com/repos/{repo}/issues/comments?per_page={max(limit // len(repos), 3)}"
            try:
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    comments = resp.json()
                    for c in comments:
                        body = c.get("body", "").strip()
                        # Clean markdown quotes and links
                        body = re.sub(r"```[\s\S]*?```", " ", body)
                        body = re.sub(r"https?://\S+", " ", body)
                        body = re.sub(r"\s+", " ", body).strip()

                        if 15 < len(body) < 350:
                            created_at = c.get("created_at", "")
                            try:
                                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                                dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except Exception:
                                dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            posts.append({
                                "text": body,
                                "source": f"GitHub ({repo})",
                                "timestamp": dt_str,
                                "created_at": dt_str,
                                "url": c.get("html_url", f"https://github.com/{repo}"),
                            })
                        if len(posts) >= limit:
                            break
                    if len(posts) >= limit:
                        break
            except Exception as e:
                logger.warning(f"GitHub fetch error ({e}).")

        if not posts:
            posts = self.load_offline_samples(source_filter="github", limit=limit)
            if not posts:
                posts = self.load_offline_samples(limit=limit)

        return posts[:limit]

    def fetch_bluesky_posts(self, query: str = "technology", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch posts from Bluesky AT Protocol public endpoint.
        Gracefully falls back to demo mode if rate-limited or blocked by Cloudflare.
        """
        url = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
        params = {"q": query or "tech", "limit": min(limit, 25)}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
        }

        posts = []
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            if resp.status_code == 200:
                raw_posts = resp.json().get("posts", [])
                for p in raw_posts:
                    record = p.get("record", {})
                    text = record.get("text", "").strip()
                    created_at = record.get("createdAt", "")
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    uri = p.get("uri", "")
                    posts.append({
                        "text": text,
                        "source": "Bluesky",
                        "timestamp": dt_str,
                        "created_at": dt_str,
                        "url": f"https://bsky.app",
                    })
                    if len(posts) >= limit:
                        break
        except Exception as e:
            logger.warning(f"Bluesky API error ({e}). Using sample stream.")

        if not posts:
            posts = self.load_offline_samples(source_filter="bluesky", limit=limit)
            if not posts:
                posts = self.load_offline_samples(limit=limit)

        return posts[:limit]

    def fetch_reddit_posts(self, subreddit: str = "technology", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch posts from Reddit. Uses PRAW if credentials exist;
        otherwise tries public JSON with seamless offline fallback.
        """
        clean_sub = re.sub(r"[^a-zA-Z0-9_]", "", subreddit) or "technology"

        if self.reddit_client_id and self.reddit_client_secret:
            try:
                import praw
                reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent,
                )
                posts = []
                sub = reddit.subreddit(clean_sub)
                for submission in sub.hot(limit=limit):
                    if submission.stickied:
                        continue
                    text = f"{submission.title}. {submission.selftext[:200]}".strip() if submission.selftext else submission.title
                    dt_str = datetime.fromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S")
                    posts.append({
                        "text": text,
                        "source": f"Reddit (r/{clean_sub})",
                        "timestamp": dt_str,
                        "created_at": dt_str,
                        "url": f"https://reddit.com{submission.permalink}",
                    })
                if posts:
                    return posts[:limit]
            except Exception as e:
                logger.warning(f"PRAW fetch failed ({e}). Falling back.")

        posts = []
        try:
            url = f"https://www.reddit.com/r/{clean_sub}/hot.json?limit={max(limit, 5)}"
            headers = {"User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) {self.reddit_user_agent}"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                children = resp.json().get("data", {}).get("children", [])
                for item in children:
                    d = item.get("data", {})
                    if d.get("stickied", False):
                        continue
                    title = d.get("title", "").strip()
                    selftext = d.get("selftext", "").strip()
                    text = f"{title}. {selftext[:200]}".strip() if selftext else title
                    created_utc = d.get("created_utc", datetime.now().timestamp())
                    dt_str = datetime.fromtimestamp(created_utc).strftime("%Y-%m-%d %H:%M:%S")
                    permalink = d.get("permalink", "")
                    post_url = f"https://reddit.com{permalink}" if permalink else f"https://reddit.com/r/{clean_sub}"

                    if text:
                        posts.append({
                            "text": text,
                            "source": f"Reddit (r/{clean_sub})",
                            "timestamp": dt_str,
                            "created_at": dt_str,
                            "url": post_url,
                        })
                    if len(posts) >= limit:
                        break
        except Exception as e:
            logger.warning(f"Reddit public JSON error: {e}")

        if not posts:
            posts = self.load_offline_samples(source_filter="reddit", limit=limit)

        return posts[:limit]

    def fetch_all_sources(self, limit_per_source: int = 3, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Aggregate posts from Hacker News, Mastodon, GitHub, and demonstration streams.
        """
        all_posts = []

        # 1. Hacker News (Live & active)
        hn_posts = self.fetch_hacker_news_posts(query=query, limit=limit_per_source)
        all_posts.extend(hn_posts)

        # 2. Mastodon (Live Fediverse)
        mastodon_posts = self.fetch_mastodon_posts(limit=limit_per_source)
        all_posts.extend(mastodon_posts)

        # 3. GitHub (Developer feedback)
        gh_posts = self.fetch_github_comments(limit=limit_per_source)
        all_posts.extend(gh_posts)

        # 4. Reddit / Demo
        reddit_posts = self.fetch_reddit_posts(subreddit=query or "technology", limit=limit_per_source)
        all_posts.extend(reddit_posts)

        if len(all_posts) < 4:
            all_posts.extend(self.load_offline_samples(limit=10))

        return all_posts


# Global fetcher instance
fetcher = SocialMediaFetcher()


def fetch_social_posts(
    source: str = "all",
    limit: int = 10,
    subreddit: str = "technology",
    query: Optional[str] = None,
    video_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Main function to fetch social media posts by source.
    Allowed sources: 'hackernews', 'mastodon', 'github', 'bluesky', 'reddit', 'all'.
    """
    src = source.lower().strip()
    if src in ("hackernews", "hacker news", "hn"):
        return fetcher.fetch_hacker_news_posts(query=query or subreddit, limit=limit)
    elif src == "mastodon":
        return fetcher.fetch_mastodon_posts(limit=limit)
    elif src in ("github", "git"):
        return fetcher.fetch_github_comments(limit=limit)
    elif src == "bluesky":
        return fetcher.fetch_bluesky_posts(query=query or subreddit or "technology", limit=limit)
    elif src == "reddit":
        return fetcher.fetch_reddit_posts(subreddit=subreddit, limit=limit)
    else:
        per_src = max(limit // 4, 3)
        return fetcher.fetch_all_sources(limit_per_source=per_src, query=query or subreddit)[:limit]


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Testing SocialMediaFetcher with free public APIs...")
    hn = fetch_social_posts(source="hackernews", limit=2)
    print(f"Hacker News ({len(hn)} items):", [p["text"][:60] for p in hn])

    masto = fetch_social_posts(source="mastodon", limit=2)
    print(f"Mastodon ({len(masto)} items):", [p["text"][:60] for p in masto])

    gh = fetch_social_posts(source="github", limit=2)
    print(f"GitHub ({len(gh)} items):", [p["text"][:60] for p in gh])
