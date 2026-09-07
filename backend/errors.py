"""Errors shared across the backend layers.

Kept in its own module so `models` (Bedrock generation) and `rag` (Bedrock
embeddings + the vector store) can raise the same type without importing each
other.
"""


class ModelBackendError(RuntimeError):
    """A dependency this answer needs was unreachable or refused the call.

    Raised at the boundary of every external AI call — Bedrock text generation,
    Bedrock embeddings, and the Chroma vector store — so callers handle one type
    instead of the full spread of botocore/chromadb exceptions.

    It deliberately means "the service is down or misconfigured", never "the
    knowledge base has no answer for this". Collapsing the two is what made a
    total outage look like an ordinary refusal: the visitor was told the
    assistant could help with anything on the site, while every request was in
    fact failing.

    Subclasses RuntimeError so pre-existing broad handlers keep working.
    """
