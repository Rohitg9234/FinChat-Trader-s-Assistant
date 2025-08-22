import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import uuid
from typing import List, Dict, Any


import re
import os
import pickle
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class EmbeddingStore:
    def __init__(self, cache_path: str = "vector_cache.pkl", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.cache_path = cache_path
        self.model = SentenceTransformer(model_name)

        # Try to load cache if exists
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self._ids = data["ids"]
            self._texts = data["texts"]
            self._embeddings = data["embeddings"]
        else:
            self._ids: List[str] = []
            self._texts: List[str] = []
            self._embeddings = np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def _save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(
                {"ids": self._ids, "texts": self._texts, "embeddings": self._embeddings},
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def _chunk_text(self, text: str, words_per_chunk: int = 250) -> List[str]:
        """Split text into ~250-word chunks."""
        words = re.findall(r"\S+", text.strip())
        return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    def add_text(self, text: str) -> List[str]:
        """
        Break paragraph into 250-word chunks, embed, and store.
        Returns list of IDs (one per chunk).
        """
        chunks = self._chunk_text(text, words_per_chunk=250)
        if not chunks:
            return []

        # embed all chunks in one call
        embeddings = self._embed(chunks)

        # append to memory
        if self._embeddings.size == 0:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        assigned_ids = []
        for chunk in chunks:
            cid = str(uuid.uuid4())
            self._ids.append(cid)
            self._texts.append(chunk)
            assigned_ids.append(cid)

        # save automatically
        self._save_cache()

        return assigned_ids

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if len(self._ids) == 0:
            return []
        q = self._embed([query_text])[0]
        scores = (self._embeddings @ q).astype(float)
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {"id": self._ids[i], "text": self._texts[i], "score": float(scores[i])}
            for i in top_idx
        ]


    # ---------- persistence ----------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"ids": self._ids, "texts": self._texts, "embeddings": self._embeddings},
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    @staticmethod
    def load(path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> "EmbeddingStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = EmbeddingStore(model_name=model_name)
        store._ids = data["ids"]
        store._texts = data["texts"]
        store._embeddings = data["embeddings"]
        return store


# ----------- demo -----------
if __name__ == "__main__":
    store = EmbeddingStore()

    # just pass text, nothing else
    store.add_text("The Eiffel Tower is in Paris.")
    store.add_text("Masala dosa is a South Indian dish.")
    store.add_text("The Taj Mahal is in Agra.")
    store.add_text("Cricket is popular in India.")

    # query
    hits = store.query("Famous landmark in France", top_k=2)
    for h in hits:
        print(h)
