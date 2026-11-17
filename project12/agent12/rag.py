def search(query: str, top_k: int) -> list[dict]:
    """Return a list like [{"source": "doc.md", "text": "..."}].

    TODO: un real RAG search ingae connect pannu, example:
        from my_rag_project.retriever import retrieve
        hits = retrieve(query, k=top_k)
        return [{"source": h.source, "text": h.text} for h in hits]
    """
    return [{"source": "stub.md", "text": f"(stub) RAG innum connect aagala. query: {query}"}]
