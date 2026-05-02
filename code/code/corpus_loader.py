"""Load and chunk local markdown corpus for retrieval."""

from __future__ import annotations

import re
from pathlib import Path

from utils import CorpusChunk, normalize_text, stable_chunk_id


def _parse_frontmatter(raw: str) -> tuple[dict[str, object], str]:
    """Split YAML-style frontmatter and return metadata + body text."""
    meta: dict[str, object] = {"title": "", "breadcrumbs": ()}
    if not raw.startswith("---"):
        return meta, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return meta, raw
    fm, body = parts[1], parts[2]
    title_m = re.search(r'(?m)^title:\s*"(.*)"\s*$', fm)
    if title_m:
        meta["title"] = title_m.group(1).strip()
    crumbs: list[str] = []
    bc_block = re.search(r"(?ms)^breadcrumbs:\s*\n((?:\s*-\s*.*\n?)+)", fm)
    if bc_block:
        for line in bc_block.group(1).splitlines():
            item = re.match(r'\s*-\s*"(.*)"\s*$', line.strip())
            if item:
                crumbs.append(item.group(1).strip())
    meta["breadcrumbs"] = tuple(crumbs)
    return meta, body.strip()


def _strip_frontmatter(raw: str) -> str:
    meta, body = _parse_frontmatter(raw)
    return body


def _extract_heading(block: str) -> str:
    for line in block.splitlines():
        if line.strip().startswith("#"):
            return line.strip("# ").strip()[:120]
    return "general"


def _split_blocks(raw_body: str) -> list[str]:
    text = raw_body
    head_split = re.split(r"\n(?=#)", text)
    blocks: list[str] = []
    for chunk in head_split:
        for paragraph in re.split(r"\n\s*\n", chunk):
            candidate = paragraph.strip()
            if candidate:
                blocks.append(candidate)
    return blocks


def _chunk_text(block: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    chunks: list[str] = []
    start = 0
    while start < len(block):
        end = min(len(block), start + max_chars)
        chunks.append(block[start:end].strip())
        if end == len(block):
            break
        start = max(0, end - overlap)
    return chunks


def _company_from_path(path: Path, data_root: Path) -> str:
    try:
        rel = path.relative_to(data_root)
        return normalize_text(rel.parts[0]) if rel.parts else "none"
    except ValueError:
        return "none"


def _product_hints(path: Path, data_root: Path) -> tuple[str, ...]:
    try:
        rel = path.relative_to(data_root)
    except ValueError:
        rel = path
    hints = [normalize_text(part.replace(".md", "").replace("-", " ")) for part in rel.parts[1:]]
    return tuple(hints)


def _slug_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^\d+-", "", stem)
    stem = re.sub(r"[%].*", "", stem)
    return normalize_text(stem.replace("-", " ")).replace(" ", "_")[:80]


def load_corpus_chunks(data_root: Path) -> list[CorpusChunk]:
    """Load markdown corpus and return deterministic chunk list."""
    md_files = sorted(data_root.rglob("*.md"), key=lambda p: str(p).lower())
    chunks: list[CorpusChunk] = []
    for file_path in md_files:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        meta, body = _parse_frontmatter(raw)
        title_str = str(meta.get("title") or "")
        breadcrumbs = meta.get("breadcrumbs")
        if not isinstance(breadcrumbs, tuple):
            breadcrumbs = ()
        blocks = _split_blocks(body)
        chunk_counter = 0
        for block in blocks:
            heading = _extract_heading(block)
            for part in _chunk_text(block):
                if not part:
                    continue
                source = str(file_path.as_posix())
                chunk_id = stable_chunk_id(source, chunk_counter)
                chunks.append(
                    CorpusChunk(
                        chunk_id=chunk_id,
                        text=part,
                        source_path=source,
                        company=_company_from_path(file_path, data_root),
                        product_hints=_product_hints(file_path, data_root),
                        heading=heading,
                        breadcrumbs=breadcrumbs,
                        article_title=title_str,
                    )
                )
                chunk_counter += 1
    return chunks
