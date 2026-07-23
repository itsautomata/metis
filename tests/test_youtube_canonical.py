"""every youtube URL form canonicalizes to one video id and one dedup key."""

import pytest

from metis.ingest.extract import (
    _canonical_youtube_url,
    canonical_youtube_id,
    is_youtube,
)

ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={ID}",
    f"https://youtube.com/watch?v={ID}",
    f"https://m.youtube.com/watch?v={ID}",
    f"https://music.youtube.com/watch?v={ID}",
    f"https://youtu.be/{ID}",
    f"https://www.youtube.com/shorts/{ID}",
    f"https://www.youtube.com/embed/{ID}",
    f"https://www.youtube.com/live/{ID}",
    f"https://www.youtube.com/v/{ID}",
    f"https://www.youtube.com/watch?v={ID}&list=PLxyz&t=30s&si=abc",
    f"https://youtu.be/{ID}?t=42",
    f"https://www.youtube.com/watch?v={ID}#t=1m",
    f"https://www.youtube.com/watch\\?v\\={ID}",  # shell-escaped paste (single backslashes)
    f"   https://youtu.be/{ID}   ",               # surrounding whitespace
])
def test_all_forms_canonicalize_to_one_id(url):
    assert canonical_youtube_id(url) == ID
    assert is_youtube(url) is True
    assert _canonical_youtube_url(url) == f"https://www.youtube.com/watch?v={ID}"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/playlist?list=PLxyz",
    "https://www.youtube.com/@channel",
    "https://www.youtube.com/watch?v=tooShort",
    "https://example.com/watch?v=dQw4w9WgXcQ",   # right shape, wrong host
    "https://vimeo.com/12345",
    "notaurl",
])
def test_non_video_urls_are_not_youtube(url):
    assert canonical_youtube_id(url) is None
    assert is_youtube(url) is False


def test_dedup_key_is_identical_across_forms():
    a = _canonical_youtube_url(f"https://youtu.be/{ID}?t=10")
    b = _canonical_youtube_url(f"https://www.youtube.com/watch?v={ID}&list=PLx")
    assert a == b  # both forms collapse to one dedup key, so one video is one note
