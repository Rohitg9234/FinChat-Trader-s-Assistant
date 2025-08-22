import re
import yt_dlp

def list_channel_videos(channel_url: str, limit: int | None = None):
    """
    Return a list of videos from a YouTube channel URL.
    Each item = {"id": "...", "url": "...", "title": "..."}.

    Args:
        channel_url (str): The YouTube channel URL
        limit (int|None): Max number of videos to fetch (None = all available)

    Returns:
        list[dict]
    """
    def _extract(url, limit=None):
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": True,   # only metadata
            "noplaylist": False,
        }
        if limit:
            ydl_opts["playlistend"] = limit
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = info.get("entries") or []
        vids = []
        for e in entries:
            vid = e.get("id") or e.get("url") or ""
            # Normalize to 11-char watch ID
            if len(vid) != 11 and "watch?v=" in str(vid):
                m = re.search(r"v=([A-Za-z0-9_-]{11})", str(vid))
                if m:
                    vid = m.group(1)
            if isinstance(vid, str) and len(vid) == 11:
                vids.append({
                    "id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": e.get("title") or ""
                })
            if limit and len(vids) >= limit:
                break
        return vids

    # 1) Try channel URL directly
    vids = []
    try:
        vids = _extract(channel_url, limit)
    except Exception:
        pass
    if vids:
        return vids

    # 2) Try /videos tab
    try:
        vids = _extract(channel_url.rstrip("/") + "/videos", limit)
    except Exception:
        pass
    if vids:
        return vids

    # 3) Try uploads playlist (UU + channel_id[2:])
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "extract_flat": True}) as ydl:
            info = ydl.extract_info(channel_url, download=False)
        chan_id = info.get("channel_id") or info.get("uploader_id")
        if chan_id and chan_id.startswith("UC") and len(chan_id) > 2:
            uploads_url = f"https://www.youtube.com/playlist?list=UU{chan_id[2:]}"
            vids = _extract(uploads_url, limit)
            if vids:
                return vids
    except Exception:
        pass

    # 4) Last resort: channel /playlists page
    try:
        vids = _extract(channel_url.rstrip("/") + "/playlists", limit)
        return vids
    except Exception:
        return []
if __name__ == "__main__":
    videos = list_channel_videos("https://www.youtube.com/@tradingwithrayner", limit=20)
    for v in videos:
        print(v["url"], "-", v["title"])
