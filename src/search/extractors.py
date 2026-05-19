"""Per-provider citation extraction (paper §B.5)."""
from __future__ import annotations

import re
from typing import NamedTuple


class Citation(NamedTuple):
    url: str
    cited_sentence: str
    order: int


_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "Dr.", "Mr.", "Mrs.", "Ms.", "U.S.", "U.K.", "vs.",
)
_URL_RE          = re.compile(r"https?://[^\s)\]]+")
_GPT_LINK_RE     = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_GROK_LINK_RE    = re.compile(r"\[\[(\d+)\]\]\((https?://[^)]+)\)")
_CITATION_RE     = re.compile(r"\[(\d+)\]")
_NUMBERED_LIST_RE = re.compile(r"\d+\.\s")
_CHROME_CHARS    = " \t\n*\"'`()“”‘’"
_SENTENCE_COUNT  = 2


def _url_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _URL_RE.finditer(text)]


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def _is_abbrev_end(text: str, dot_pos: int) -> bool:
    lower = text.lower()
    for abbr in _ABBREVIATIONS:
        a = abbr.lower()
        for offset in range(len(a)):
            if a[offset] != ".":
                continue
            start = dot_pos - offset
            end = start + len(a)
            if 0 <= start and end <= len(text) and lower[start:end] == a:
                return True
    return False


def _is_sentence_end(text: str, pos: int, url_spans: list[tuple[int, int]]) -> bool:
    if pos < 0 or pos >= len(text):
        return False
    ch = text[pos]
    if ch not in ".!?":
        return False
    if _in_spans(pos, url_spans):
        return False
    if ch == "." and _is_abbrev_end(text, pos):
        return False
    return True


def _is_md_boundary(text: str, pos: int) -> bool:
    if pos < 0 or pos >= len(text) or text[pos] != "\n":
        return False
    if pos + 1 < len(text) and text[pos + 1] == "\n":
        return True
    j = pos + 1
    while j < len(text) and text[j] in " \t":
        j += 1
    rest = text[j:j + 4]
    if rest.startswith("- ") or rest.startswith("* ") or rest.startswith("### "):
        return True
    if _NUMBERED_LIST_RE.match(rest):
        return True
    return False


def _clean(s: str) -> str:
    s = s.strip(_CHROME_CHARS)
    while True:
        m = re.match(r"^\s*\[\d+\]\s*", s)
        if not m:
            break
        s = s[m.end():]
    return s.strip(_CHROME_CHARS)


def _extract_last_sentences(text: str, n: int = _SENTENCE_COUNT) -> list[str]:
    stripped = text.rstrip()
    while stripped and stripped[-1] in _CHROME_CHARS:
        stripped = stripped[:-1].rstrip()
    if not stripped:
        return []

    url_spans = _url_spans(stripped)
    boundaries: list[int] = [len(stripped)]
    i = len(stripped) - 2
    while i >= 0 and len(boundaries) <= n:
        candidate = None
        if _is_sentence_end(stripped, i, url_spans):
            candidate = i + 1
        elif _is_md_boundary(stripped, i):
            candidate = i + (2 if (i + 1 < len(stripped) and stripped[i + 1] == "\n") else 1)
        if candidate is not None:
            boundaries.append(candidate)
            if len(boundaries) > n:
                break
        i -= 1
    if len(boundaries) <= n:
        boundaries.append(0)

    out = []
    for a, b in zip(boundaries[1:], boundaries[:-1]):
        piece = _clean(stripped[a:b])
        if piece:
            out.append(piece)
    out.reverse()
    return out


def _join_sentences(sents: list[str]) -> str:
    return " ".join(sents).strip()


def _replace_gpt_links(text: str, annotations: list[dict]) -> str:
    """OpenAI / xAI markdown links ``[text](url)`` → ``[N]`` numbered."""
    url_to_num: dict[str, int] = {}
    for idx, ann in enumerate(annotations, start=1):
        url = ann.get("url", "")
        if url and url not in url_to_num:
            url_to_num[url] = idx
    next_num = max(url_to_num.values(), default=0)

    def _r(m: re.Match) -> str:
        nonlocal next_num
        url = m.group(2)
        if url not in url_to_num:
            next_num += 1
            url_to_num[url] = next_num
        return f"[{url_to_num[url]}]"

    return _GPT_LINK_RE.sub(_r, text), url_to_num


def _replace_grok_links(text: str) -> tuple[str, dict[str, int]]:
    url_to_num: dict[str, int] = {}

    def _r(m: re.Match) -> str:
        n = int(m.group(1))
        url_to_num.setdefault(m.group(2), n)
        return f"[{n}]"

    return _GROK_LINK_RE.sub(_r, text), url_to_num


def _strip_sonar_url_list(text: str) -> str:
    return re.split(r"\n\s*\[\d+\]\s+https?://", text, maxsplit=1)[0]


def _find_citations(text: str) -> list[tuple[int, int, int]]:
    """(N, start, end) tuples for each ``[N]`` marker."""
    return [(int(m.group(1)), m.start(), m.end()) for m in _CITATION_RE.finditer(text)]


def _rows_from_numbered_text(text: str, num_to_url: dict[int, str]) -> list[Citation]:
    rows: list[Citation] = []
    cits = _find_citations(text)
    i, n = 0, len(cits)
    while i < n:
        j = i
        # Cluster: consecutive markers with no chars between them.
        while j + 1 < n and cits[j + 1][1] == cits[j][2]:
            j += 1
        sentence = _join_sentences(_extract_last_sentences(text[:cits[i][1]]))
        for k in range(i, j + 1):
            num = cits[k][0]
            url = (num_to_url.get(num) or "").strip()
            if url:
                rows.append(Citation(url=url, cited_sentence=sentence, order=num))
        i = j + 1
    return rows


# ──────────────────────────── per-provider ────────────────────────────────


def extract_openai(raw: str, meta: dict) -> list[Citation]:
    annotations = (meta or {}).get("annotations") or []
    numbered, url_to_num = _replace_gpt_links(raw or "", annotations)
    num_to_url = {n: u for u, n in url_to_num.items()}
    return _rows_from_numbered_text(numbered, num_to_url)


def extract_grok(raw: str, meta: dict) -> list[Citation]:
    annotations = (meta or {}).get("annotations") or []
    numbered, url_to_num = _replace_grok_links(raw or "")
    if not url_to_num:
        # Some Grok responses return positions via annotations (no inline markdown);
        # fall back to numbering annotations the same way as OpenAI.
        numbered, url_to_num = _replace_gpt_links(raw or "", annotations)
    num_to_url = {n: u for u, n in url_to_num.items()}
    return _rows_from_numbered_text(numbered, num_to_url)


def extract_perplexity(raw: str, meta: dict) -> list[Citation]:
    body = _strip_sonar_url_list(raw or "")
    raw_cits = (meta or {}).get("citations") or {}
    if isinstance(raw_cits, list):
        num_to_url = {i + 1: u for i, u in enumerate(raw_cits)}
    elif isinstance(raw_cits, dict):
        num_to_url = {int(k): v for k, v in raw_cits.items()}
    else:
        num_to_url = {}
    if not num_to_url:
        for i, sr in enumerate((meta or {}).get("search_results") or [], start=1):
            num_to_url[i] = (sr.get("url") or "") if isinstance(sr, dict) else ""
    return _rows_from_numbered_text(body, num_to_url)


def extract_anthropic(raw: str, meta: dict) -> list[Citation]:
    """Anthropic returns native citation blocks with explicit ``cited_text``."""
    del raw
    out: list[Citation] = []
    cits = (meta or {}).get("native_citations") or (meta or {}).get("text_citations") or []
    for idx, cit in enumerate(cits, start=1):
        url = (cit.get("url") or "").strip()
        if not url:
            continue
        text = cit.get("cited_text") or cit.get("text") or ""
        out.append(Citation(
            url=url,
            cited_sentence=_join_sentences(_extract_last_sentences(text)),
            order=idx,
        ))
    return out


def extract_gemini(raw: str, meta: dict) -> list[Citation]:
    """Gemini returns ``grounding_supports[].text`` with ``chunk_indices`` into
    ``grounding_chunks[].uri``. One row emitted per (support, chunk_index)."""
    del raw
    chunks   = (meta or {}).get("grounding_chunks") or []
    supports = (meta or {}).get("grounding_supports") or []
    out: list[Citation] = []
    num = 1
    for sup in supports:
        sentence = _join_sentences(_extract_last_sentences(sup.get("text", "")))
        for i in (sup.get("chunk_indices") or sup.get("grounding_chunk_indices") or []):
            if 0 <= i < len(chunks):
                url = (chunks[i].get("uri") or "").strip()
                if url:
                    out.append(Citation(url=url, cited_sentence=sentence, order=num))
            num += 1
    return out


_DISPATCH = {
    "openai":     extract_openai,
    "anthropic":  extract_anthropic,
    "google":     extract_gemini,
    "xai":        extract_grok,
    "perplexity": extract_perplexity,
}


def extract(provider: str, raw_response: str, api_metadata: dict) -> list[Citation]:
    fn = _DISPATCH.get(provider)
    if not fn:
        raise ValueError(f"unknown provider: {provider!r}")
    return fn(raw_response, api_metadata or {})


__all__ = ["Citation", "extract"]
