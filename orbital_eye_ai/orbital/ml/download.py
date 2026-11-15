"""Download the trained checkpoint from a private link at start-up (hosted demo).

The weights are derived from the Duke DME dataset, whose terms prohibit
redistributing the images, so they are kept out of the public repository and
fetched from a secret link (Space secret / environment variable MODEL_URL).
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DRIVE_PATTERNS = [
    r"drive\.google\.com/file/d/([A-Za-z0-9_-]{10,})",
    r"drive\.google\.com/open\?id=([A-Za-z0-9_-]{10,})",
    r"[?&]id=([A-Za-z0-9_-]{10,})",
]


def google_drive_id(url: str) -> Optional[str]:
    if "google.com" not in url:
        return None
    for pattern in _DRIVE_PATTERNS:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def direct_url(url: str) -> str:
    """Turn a Google Drive share link into a direct download URL (skips the large-file warning page)."""
    file_id = google_drive_id(url)
    if file_id:
        return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return url


def ensure_model(model_path: Path, url: str, timeout: int = 120) -> bool:
    """Download the checkpoint if it is missing and a URL is configured. Returns True if the file exists."""
    model_path = Path(model_path)
    if model_path.exists():
        return True
    if not url:
        return False
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(direct_url(url), headers={"User-Agent": "orbital-eye-ai"})
        with urllib.request.urlopen(request, timeout=timeout) as response, \
                tempfile.NamedTemporaryFile(dir=str(model_path.parent), delete=False) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
        with open(tmp_path, "rb") as f:
            magic = f.read(2)
        if magic != b"PK":  # torch.save files are zip archives; an HTML page means a bad/unshared link
            tmp_path.unlink(missing_ok=True)
            log.error("MODEL_URL did not return a PyTorch checkpoint. Is the Drive file shared as "
                      "'Anyone with the link'?")
            return False
        tmp_path.replace(model_path)
        log.info("Downloaded model checkpoint (%.1f MB)", model_path.stat().st_size / 1e6)
        return True
    except Exception as exc:  # network errors must not stop the app; it falls back to the heuristic
        log.error("Could not download model checkpoint: %s", exc)
        return False
