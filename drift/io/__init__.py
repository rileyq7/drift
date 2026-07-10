"""drift/io — Reading from files and URLs.

v0.2 stubs. Real implementations TBD.

Trust model: these helpers do NO path confinement — `read`/`write` can touch
any path the process can, and `fetch_url` any host it can reach. A .drift
program (and any MCP caller of `drift_run`, which is arbitrary code execution
by design) therefore has full host file/network access. Sandbox at the process
level if you run untrusted Drift.
"""
import os
from pathlib import Path

# Cap fetch_url response size so a hostile/huge URL can't exhaust memory.
# Override via DRIFT_MAX_FETCH_BYTES.
try:
    _MAX_FETCH_BYTES = int(os.environ.get("DRIFT_MAX_FETCH_BYTES", str(10 * 1024 * 1024)))
except ValueError:
    _MAX_FETCH_BYTES = 10 * 1024 * 1024


def read(path: str) -> str:
    """Read a text file from disk."""
    return Path(path).read_text()


def write(path: str, content: str) -> None:
    """Write a text file to disk, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


async def fetch_url(url: str) -> str:
    """Fetch text content from a URL (capped at DRIFT_MAX_FETCH_BYTES)."""
    import httpx
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, timeout=30.0) as resp:
            resp.raise_for_status()
            chunks = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > _MAX_FETCH_BYTES:
                    raise RuntimeError(
                        f"fetch_url response exceeded {_MAX_FETCH_BYTES} bytes"
                    )
                chunks.append(chunk)
            encoding = resp.encoding or "utf-8"
            return b"".join(chunks).decode(encoding, errors="replace")


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
