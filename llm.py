# pip install mistralai
import os
from typing import Iterable, List, Optional, Generator, Union
from mistralai import Mistral

DEFAULT_MODEL = "mistral-small-latest"

def _format_context(chunks: Iterable[str], max_chunks: int = 12) -> str:
    """Join retrieved chunks with clear delimiters and mild deduping."""
    seen = set()
    joined: List[str] = []
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue
        key = c[:200]  # shallow dedupe by prefix
        if key in seen:
            continue
        seen.add(key)
        joined.append(c)
        if len(joined) >= max_chunks:
            break
    if not joined:
        return "No additional context provided."
    return "\n\n---\n\n".join(joined)

def answer_with_tone(
    query: str,
    chunks: Iterable[str],
    tone: str = "concise, friendly",
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 1024,
    stream: bool = False,
    system_preamble: Optional[str] = None,
    cite_sources: bool = True,
) -> Union[str, Generator[str, None, None]]:
    """
    Use Mistral chat API to answer `query` using `chunks` as retrieval context,
    phrased in the requested `tone`.

    Parameters
    ----------
    query : str
        The user's question.
    chunks : Iterable[str]
        Retrieved context snippets (docs, notes, passages).
    tone : str
        Desired voice/style, e.g., "witty and informal", "formal and academic".
    api_key : Optional[str]
        If None, will read from env var MISTRAL_API_KEY.
    model : str
        Mistral model name. Defaults to 'mistral-large-latest'.
    temperature : float
        Decoding temperature.
    max_tokens : int
        Max tokens for the completion.
    stream : bool
        If True, yields incremental text chunks (generator).
    system_preamble : Optional[str]
        Extra system guidance merged with the default system message.
    cite_sources : bool
        If True, the assistant will include lightweight inline citations like [S1], [S2] tied to chunk indices.

    Returns
    -------
    str or generator of str
        Final answer string if stream=False, else a generator yielding deltas.
    """
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is required (pass api_key= or set env var).")

    context_str = _format_context(chunks)

    # Build a compact instruction that binds tone + task + use of chunks.
    default_system = f"""You are a helpful assistant. Match the user's requested tone exactly: {tone}.
- Use ONLY the provided context when possible.
- Prefer clear, direct phrasing. Keep the answer tightly scoped to the query.
- If lists are helpful, use short bullets.
"""

    if system_preamble:
        default_system = system_preamble.strip() + "\n\n" + default_system

    # We show numbered chunks to enable lightweight citations.
    numbered_context = []
    for i, c in enumerate(context_str.split("\n\n---\n\n"), start=1):
        numbered_context.append(f"[S{i}] {c.strip()}")
    numbered_context_str = "\n\n".join(numbered_context) if numbered_context else context_str

    user_prompt = f"""Answer the user's query using the context below and the requested tone.
If the context lacks the answer, say so briefly and then answer using general knowledge (clearly marked).

# Query
{query.strip()}

# Context Chunks
{numbered_context_str}

Note: No need to mention the context chunks in the answer., act as a expert in the field.
"""

    messages = [
        {"role": "system", "content": default_system},
        {"role": "user", "content": user_prompt},
    ]

    client = Mistral(api_key=api_key)
    if not stream:
        with client:
            resp = client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        return resp.choices[0].message.content

    # Streaming branch: yield text deltas as they arrive
    def _stream() -> Generator[str, None, None]:
        with client:
            for event in client.chat.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if event.data and hasattr(event.data, "delta") and event.data.delta:
                    yield event.data.delta

    return _stream()

# -----------------------
# Example usage (non-streaming)
if __name__ == "__main__":
    example_chunks = [
        "Transformers process sequences in parallel using self-attention instead of recurrence.",
        "Self-attention layers compute pairwise token interactions to build contextualized representations.",
    ]
    reply = answer_with_tone(
        query="Explain transformers in 2 sentences.",
        chunks=example_chunks,
        tone="concise and didactic",
        temperature=0.6,
        max_tokens=256,
        stream=False,
    )
    print(reply)

    # Example (streaming)
    # for piece in answer_with_tone("Key ideas of self-attention?", example_chunks, tone="witty", stream=True):
    #     print(piece, end="", flush=True)
