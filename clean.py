import re

# A compact, extensible stopword list (tweak as needed)
STOPWORDS = {
    "a","an","and","are","as","at","be","but","by","for","from","has","have","had","how",
    "i","in","into","is","it","its","let","me","my","of","on","or","so","than","that",
    "the","their","them","then","there","these","they","this","those","to","us","was",
    "were","what","when","where","which","who","whom","why","will","with","would","you",
    "your","please","can","could","should","need","want","looking","about","give","show",
    "find","tell","help","get","like","some","any","just","also","around","over","up",
    "down","out","off","more","most","least","maybe","perhaps","kind","sort"
}

# Common search operators we want to preserve (add more if you use them)
OPERATOR_PREFIXES = ("site:", "inurl:", "intitle:", "filetype:", "lang:", "before:", "after:")
LOGICAL_TOKENS = {"AND", "OR", "NOT", "-"}

def clean_query(
    query: str,
    *,
    domain_terms: set[str] | None = None,   # words you ALWAYS want to keep (e.g., brand/feature nouns)
    keep_operators: bool = True,
    min_token_len: int = 2,                 # drop very short tokens unless numeric
    dedupe: bool = True
) -> str:
    """
    Remove irrelevant words while preserving quoted phrases and search operators.
    Returns a cleaned query string.
    """
    if not query or not isinstance(query, str):
        return ""

    domain_terms = {t.lower() for t in (domain_terms or set())}

    # 1) Extract and protect quoted phrases (keep them verbatim)
    quoted = re.findall(r'"([^"]+)"', query)
    protected_phrases = [f'"{q.strip()}"' for q in quoted]
    query_wo_quotes = re.sub(r'"[^"]+"', " ", query)

    # 2) Tokenize the rest (simple split respecting punctuation)
    #    Keep colon for operators; strip other punctuation except + and # (useful in tech terms)
    rough_tokens = re.split(r"\s+", query_wo_quotes.strip())

    cleaned_tokens = []
    seen = set()

    def should_keep(token: str) -> bool:
        if not token:
            return False
        # Keep if it’s a known logical token
        if token in LOGICAL_TOKENS:
            return True
        # Keep operators like site:example.com
        if keep_operators and token.lower().startswith(OPERATOR_PREFIXES):
            return True
        # Keep hyphen as NOT operator only if it prefixes a real word e.g., -free
        if token.startswith("-"):
            base = token[1:]
            return bool(base) and should_keep(base)

        # Normalize for stopword/domain checks (strip surrounding punctuation except : + #)
        norm = re.sub(r"^[^\w#+:]+|[^\w#+:]+$", "", token)
        if not norm:
            return False

        # Keep if in domain terms
        if norm.lower() in domain_terms:
            return True

        # Drop if stopword
        if norm.lower() in STOPWORDS:
            return False

        # Drop super short tokens (unless numeric like "4k" or "gpt-4")
        if len(norm) < min_token_len and not re.search(r"\d", norm):
            return False

        return True

    # 3) Clean each token
    for tok in rough_tokens:
        # Preserve raw if operator token; else strip surrounding punctuation
        keep_raw = keep_operators and tok.lower().startswith(OPERATOR_PREFIXES)
        tok_clean = tok if keep_raw else re.sub(r"^[^\w#+:-]+|[^\w#+:-]+$", "", tok)

        # Uppercase logical tokens consistently
        if tok_clean.lower() in {t.lower() for t in LOGICAL_TOKENS}:
            tok_clean = tok_clean.upper()

        if should_keep(tok_clean):
            key = tok_clean.lower()
            if not dedupe or key not in seen:
                cleaned_tokens.append(tok_clean)
                seen.add(key)

    # 4) Reattach protected quoted phrases (dedupe applies here too)
    for phrase in protected_phrases:
        key = phrase.lower()
        if not dedupe or key not in seen:
            cleaned_tokens.append(phrase)
            seen.add(key)

    # 5) Simple pass to remove dangling operators (e.g., leading OR, trailing AND)
    def is_logic(t): return t in LOGICAL_TOKENS
    pruned = []
    for i, t in enumerate(cleaned_tokens):
        if is_logic(t):
            if i == 0 or i == len(cleaned_tokens) - 1:
                continue
            if is_logic(cleaned_tokens[i - 1]) or is_logic(cleaned_tokens[i + 1]):
                continue
        pruned.append(t)

    # 6) Join with spaces
    return " ".join(pruned)
