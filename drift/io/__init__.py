"""drift/io — Reading from files and URLs.

v0.2 stubs. Real implementations TBD.
"""
from pathlib import Path


def read(path: str) -> str:
    """Read a text file from disk."""
    return Path(path).read_text()


def write(path: str, content: str) -> None:
    """Write a text file to disk, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


async def fetch_url(url: str) -> str:
    """Fetch text content from a URL."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.text


def load_pdf(path: str) -> str:
    """Extract text from a PDF. Requires pypdf to be installed."""
    try:
        import pypdf
    except ImportError as e:
        raise RuntimeError(
            "drift.io.load_pdf requires `pypdf` — install with `pip install pypdf`"
        ) from e
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def load_csv(path: str) -> list[dict]:
    """Read a CSV file as a list of dicts (one per row)."""
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


__all__ = ["read", "write", "fetch_url", "load_pdf", "load_csv"]
