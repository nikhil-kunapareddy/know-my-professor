"""Does reranking improve what the model is handed, and where should the cutoff go?

Two arms over one retrieval: the cosine order Pinecone returns, and the order a
cross-encoder puts the same chunks in. Because both arms hold the SAME chunks,
every judged verdict counts for both — and the 900 verdicts ``evaluation.topk``
already paid for are reused rather than re-bought.

``evaluation.topk`` is imported, never modified: its judge, its cache, and its
per-k arithmetic are the measuring instrument, and changing the instrument
between experiments would make the numbers incomparable.
"""
