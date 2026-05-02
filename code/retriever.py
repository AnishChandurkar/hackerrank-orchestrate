"""
retriever.py — Knowledge-base loader for the RAG pipeline.

Recursively loads every .md file from:
    data/hackerrank/
    data/claude/
    data/visa/

For each file, YAML frontmatter is parsed to extract ``title`` and
``breadcrumbs``.  The immediate subfolder relative to the domain folder is
captured as ``subfolder``; files sitting directly in the domain folder receive
``subfolder = "general"``.

Chunk schema
------------
{
    "domain":          str,   # "hackerrank" | "claude" | "visa"
    "subfolder":       str,   # e.g. "amazon-bedrock" | "general"
    "source_filename": str,   # file stem (no extension)
    "title":           str,   # frontmatter title, or stem as fallback
    "breadcrumbs":     list,  # frontmatter breadcrumbs, or []
    "product_area":    str,   # breadcrumbs[0] if present, else subfolder
    "text":            str,   # body text after the frontmatter block
}

Usage
-----
    from retriever import load_all_chunks
    chunks = load_all_chunks()

CLI smoke-test
--------------
    python retriever.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths — resolved relative to *this* file so the module works from any cwd
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_DATA_ROOT: Path = _REPO_ROOT / "data"

# Maps domain label -> absolute Path of domain folder
DOMAIN_FOLDERS: dict[str, Path] = {
    "hackerrank": _DATA_ROOT / "hackerrank",
    "claude":     _DATA_ROOT / "claude",
    "visa":       _DATA_ROOT / "visa",
}

# ---------------------------------------------------------------------------
# YAML frontmatter parsing
# ---------------------------------------------------------------------------

# Matches an opening "---\n", captures everything up to the closing "---",
# then an optional newline.  Works with both \n and \r\n line endings.
_FM_RE = re.compile(
    r"^---[ \t]*\r?\n(?P<fm>.*?)\r?\n---[ \t]*\r?\n?",
    re.DOTALL,
)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split *raw* into (frontmatter_dict, body_text).

    Returns ({}, raw) when no valid YAML frontmatter block is found or when
    the block fails to parse.
    """
    match = _FM_RE.match(raw)
    if not match:
        return {}, raw

    body: str = raw[match.end():]

    try:
        meta = yaml.safe_load(match.group("fm")) or {}
    except yaml.YAMLError:
        meta = {}

    if not isinstance(meta, dict):
        meta = {}

    return meta, body


# ---------------------------------------------------------------------------
# Subfolder resolution
# ---------------------------------------------------------------------------

def _resolve_subfolder(md_path: Path, domain_folder: Path) -> str:
    """Return the *immediate* subfolder name under *domain_folder*.

    If the file lives directly inside *domain_folder* (depth == 1 relative
    part, i.e. just the filename), return ``"general"``.

    Examples
    --------
    data/claude/amazon-bedrock/foo.md  ->  "amazon-bedrock"
    data/claude/foo.md                 ->  "general"
    data/hackerrank/a/b/deep.md        ->  "a"   (immediate child only)
    """
    # relative_to raises ValueError if md_path is not under domain_folder;
    # that should never happen given we rglob from the same folder.
    parts = md_path.relative_to(domain_folder).parts  # e.g. ("amazon-bedrock", "foo.md")

    if len(parts) <= 1:
        return "general"

    return parts[0]   # immediate subfolder name


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------

def _load_file(md_path: Path, domain: str, domain_folder: Path) -> dict[str, Any]:
    """Read one .md file and return a chunk dict."""
    raw: str = md_path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_frontmatter(raw)

    title: str = meta.get("title") or md_path.stem

    # breadcrumbs: must be a list of strings
    raw_bc = meta.get("breadcrumbs")
    if isinstance(raw_bc, list):
        breadcrumbs: list[str] = [str(b) for b in raw_bc]
    elif raw_bc:
        breadcrumbs = [str(raw_bc)]
    else:
        breadcrumbs = []

    subfolder: str = _resolve_subfolder(md_path, domain_folder)

    # product_area: first breadcrumb wins; fall back to subfolder
    product_area: str = breadcrumbs[0] if breadcrumbs else subfolder

    body_stripped = body.strip()
    if len(body_stripped) > 800:
        raw_text = body_stripped[:800]
        # Priority order for cut points — prefer the cleanest boundary:
        #   1. \n\n  (paragraph break — cleanest, avoids splitting lists/images)
        #   2. .\n   (end-of-sentence at a line break)
        #   3. .     (end-of-sentence mid-paragraph)
        #   4. \n    (any line break)
        #   5. hard 800-char cut (last resort)
        cut = raw_text.rfind('\n\n')
        if cut == -1:
            cut = raw_text.rfind('.\n')
            if cut != -1:
                cut += 1          # include the period
        if cut == -1:
            cut = raw_text.rfind('.')
            if cut != -1:
                cut += 1          # include the period
        if cut == -1:
            cut = raw_text.rfind('\n')
        if cut != -1:
            final_text = raw_text[:cut].strip()
        else:
            final_text = raw_text
    else:
        final_text = body_stripped

    return {
        "domain":          domain,
        "subfolder":       subfolder,
        "source_filename": md_path.stem,
        "title":           title,
        "breadcrumbs":     breadcrumbs,
        "product_area":    product_area,
        "text":            final_text,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_chunks() -> list[dict[str, Any]]:
    """Recursively load every .md file from all domain folders.

    Files that cannot be read are skipped with a warning; the rest of the
    load continues normally.
    """
    chunks: list[dict[str, Any]] = []

    for domain, domain_folder in DOMAIN_FOLDERS.items():
        if not domain_folder.exists():
            print(f"[retriever] WARNING: domain folder missing — {domain_folder}")
            continue

        for md_path in sorted(domain_folder.rglob("*.md")):
            try:
                chunk = _load_file(md_path, domain, domain_folder)

                # Filter out shallow index/navigation pages that pollute BM25 results.
                #
                # Two categories are skipped:
                #
                # 1. Title contains "FAQs" — these are top-level category landing pages
                #    (e.g. "Coding Challenges FAQs", "Manage Account FAQs") that list
                #    sub-topics but contain no direct answers.  Because they repeat
                #    high-frequency terms like "FAQs", "account", "billing" across many
                #    documents, they inflate BM25 IDF weights for those terms and
                #    consistently outrank genuinely useful articles.
                #
                # 2. Body text shorter than 200 characters — these are stub or redirect
                #    pages whose body is essentially empty after frontmatter is stripped.
                #    They add noise to the BM25 corpus without providing any answer
                #    content that could satisfy a support query.
                if "FAQs" in chunk["title"]:
                    continue
                if len(chunk["text"]) < 200:
                    continue

                chunks.append(chunk)
            except Exception as exc:  # noqa: BLE001
                print(f"[retriever] WARNING: skipping {md_path.name} — {exc}")

    return chunks


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

# LLM used downstream for answer generation: llama-3.3-70b-versatile (Groq)
# This comment documents the model so router.py / llm.py can stay in sync.

# Why filter BEFORE scoring?
# ---------------------------
# BM25 computes IDF (inverse document frequency) over the corpus it is given.
# Running it against 773 unrelated documents would dilute IDF weights: a term
# that is rare within the "claude" domain looks common across all domains,
# lowering its discriminative power.  Pre-filtering to the relevant domain
# (and optionally subfolder) gives BM25 a tighter, more coherent corpus so
# IDF scores are meaningful and the top-k results are more precise.

# Why a minimum-chunk threshold (5) for subfolder filtering?
# ----------------------------------------------------------
# If a subfolder contains fewer than 5 chunks the BM25 corpus would be tiny
# and essentially random — the query tokenises into terms that either all
# appear or none do, making the ranking meaningless.  Falling back to the
# full domain set ensures there is enough statistical context for IDF to work.
_SUBFOLDER_MIN_CHUNKS = 5


class BM25Retriever:
    """BM25Okapi index over the knowledge-base chunks.

    Construction
    ------------
    Pass a list of chunk dicts (as returned by ``load_all_chunks()``).  The
    index is built once at init time; repeated queries are cheap.

    Indexing strategy
    -----------------
    Each document fed to BM25 is:
        title + " " + title + " " + text
    Doubling the title causes BM25's TF component to count title tokens twice,
    effectively up-weighting them without requiring a custom scorer.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        try:
            from rank_bm25 import BM25Okapi  # local import keeps startup fast
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 is required: pip install rank-bm25"
            ) from exc

        self._BM25Okapi = BM25Okapi
        self._chunks: list[dict[str, Any]] = chunks

        # Build a global index so we can fall back to it when filters are off.
        # Individual filtered corpora are built on-the-fly inside query().
        self._global_index, self._global_corpus = self._build_index(chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lower-case whitespace tokenisation (good enough for BM25)."""
        return text.lower().split()

    def _make_doc(self, chunk: dict[str, Any]) -> str:
        """Return the BM25 document string for *chunk*.

        Title is repeated twice to give it higher TF weight.
        """
        title: str = chunk.get("title") or ""
        text: str  = chunk.get("text")  or ""
        return f"{title} {title} {text}"

    def _build_index(
        self, subset: list[dict[str, Any]]
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Build a BM25Okapi index over *subset* and return (index, subset)."""
        corpus = [self._tokenise(self._make_doc(c)) for c in subset]
        index  = self._BM25Okapi(corpus)
        return index, subset

    # ------------------------------------------------------------------
    # Internal: shared filter + scoring pipeline
    # ------------------------------------------------------------------

    def _ranked_results(
        self,
        text: str,
        domain: str | None,
        subfolder: str | None,
        top_k: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Core retrieval pipeline shared by query() and query_with_confidence().

        Returns a list of (score, chunk) pairs sorted by score descending,
        truncated to *top_k*.  Returns an empty list when no pool exists.

        Filtering pipeline
        ------------------
        1. Domain filter (e.g. "claude"): reduces ~749 → ~300.
           Always applied when *domain* is given — keeps IDF tight.
        2. Subfolder filter (e.g. "amazon-bedrock"): reduces ~300 → ~50.
           Only applied when the subfolder contains ≥ ``_SUBFOLDER_MIN_CHUNKS``
           chunks; otherwise we fall back to the domain-filtered set to avoid
           a corpus that is too small for meaningful IDF scoring.
        3. BM25 scoring on the filtered set.
        4. Return top_k (score, chunk) pairs, highest score first.
        """
        # --- Step 1: domain filter -------------------------------------------
        if domain is not None:
            pool = [c for c in self._chunks if c["domain"] == domain]
        else:
            pool = self._chunks  # no domain hint → search everything

        if not pool:
            return []

        # --- Step 2: subfolder filter (with minimum-chunk guard) -------------
        if subfolder is not None:
            subfolder_pool = [c for c in pool if c["subfolder"] == subfolder]
            if len(subfolder_pool) >= _SUBFOLDER_MIN_CHUNKS:
                # Large enough corpus — subfolder filter improves precision.
                pool = subfolder_pool
            # else: too few chunks in this subfolder; stay on domain pool so
            # BM25 has enough context for reliable IDF computation.

        # --- Step 3: build a BM25 index over the filtered pool ---------------
        # We rebuild per query rather than caching every (domain, subfolder)
        # pair; ~749 docs tokenise in < 50 ms, well within latency budget.
        index, corpus = self._build_index(pool)

        # --- Step 4: score and rank -------------------------------------------
        query_tokens = self._tokenise(text)
        scores: list[float] = index.get_scores(query_tokens)

        # Pair each chunk with its score, sort descending, take top_k.
        ranked = sorted(
            zip(scores, corpus),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        domain: str | None = None,
        subfolder: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant chunks for *text*.

        See ``_ranked_results`` for the filtering pipeline.
        """
        return [chunk for _, chunk in self._ranked_results(text, domain, subfolder, top_k)]

    def query_with_confidence(
        self,
        text: str,
        domain: str | None = None,
        subfolder: str | None = None,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], float]:
        """Return *(chunks, top_score)* for *text*.

        *top_score* is the highest raw BM25 score among the returned results
        (0.0 when no documents match the domain/subfolder filters).  Callers
        can use this to decide whether the retrieval result is trustworthy:

            chunks, top_score = retriever.query_with_confidence(text, domain=domain)
            if top_score < 0.3:
                force_escalate = True  # corpus has no relevant document

        Why 0.3?
        --------
        BM25Okapi scores have no fixed upper bound, but in practice a score
        below 0.3 over a domain-filtered corpus of ~250-750 documents means
        no query token appeared at a meaningful frequency in any document —
        i.e. the knowledge base contains nothing relevant to the question.
        0.3 was chosen empirically by inspecting score distributions on the
        sample support tickets; adjust if your corpus characteristics change.
        """
        ranked = self._ranked_results(text, domain, subfolder, top_k)
        if not ranked:
            return [], 0.0
        top_score: float = float(ranked[0][0])
        chunks = [chunk for _, chunk in ranked]
        return chunks, top_score


# ---------------------------------------------------------------------------
# CLI smoke-test:  python retriever.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_chunks = load_all_chunks()
    for chunk in all_chunks[:5]:
        print(len(chunk['text']), chunk['text'][-50:])
    
    print(f"Loaded {len(all_chunks)} chunk(s) across {len(DOMAIN_FOLDERS)} domain(s).\n")

    for chunk in all_chunks[:10]:
        print(
            f"  [{chunk['domain']:>10}]  {chunk['subfolder']:<28}"
            f"  {chunk['source_filename'][:40]:<40}"
            f"  title={chunk['title']!r:.50}"
        )
    if len(all_chunks) > 10:
        print(f"  … and {len(all_chunks) - 10} more.\n")

    # BM25 smoke-test
    retriever = BM25Retriever(all_chunks)
    results = retriever.query(
        "how do I get access to Claude in Amazon Bedrock",
        domain="claude",
        subfolder="amazon-bedrock",
        top_k=3,
    )
    print("BM25 query -> 'Claude in Amazon Bedrock' (domain=claude, subfolder=amazon-bedrock):")
    for r in results:
        print(f"  [{r['subfolder']}]  {r['title']!r}")
    print("Test 1 done")
    
    # Test 2
    results2 = retriever.query(
        "how do I reschedule my assessment",
        domain="hackerrank",
        top_k=3,
    )
    print("\nBM25 query -> 'reschedule assessment' (domain=hackerrank):")
    for r in results2:
        print(f"  [{r['subfolder']}]  {r['title']!r}")

    print("Test 2 done")

    # Test 3
    results3 = retriever.query(
        "dispute a charge",
        domain="visa",
        top_k=3,
    )
    print("\nBM25 query -> 'dispute a charge' (domain=visa):")
    for r in results3:
        print(f"  [{r['subfolder']}]  {r['title']!r}")

    print("Test 3 done")
