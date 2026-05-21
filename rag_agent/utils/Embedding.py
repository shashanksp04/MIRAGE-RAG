import torch
from sentence_transformers import SentenceTransformer
from typing import List


class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None"):
        if device == "None":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(
            input, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

    def embed_one(self, text: str) -> List[float]:
        """Embed a single string and return a flat float list."""
        return self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()[0]

    @property
    def vector_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
