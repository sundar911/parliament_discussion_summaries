from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from ..config.models import TopicSettings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TopicAssignment:
    topic_id: int
    label: str
    score: float
    sentences: List[str]


class TopicModel:
    """Simple topic discovery using embeddings + KMeans (placeholder for BERTopic)."""

    def __init__(self, settings: TopicSettings) -> None:
        self.settings = settings
        self._encoder: Optional[SentenceTransformer] = None
        self._labels: Dict[int, str] = {}

    def _load_encoder(self) -> None:
        if self._encoder is None:
            logger.info("Loading topic embedding model %s", self.settings.embedding_model)
            self._encoder = SentenceTransformer(self.settings.embedding_model)

    def fit(self, documents: Sequence[str]) -> List[TopicAssignment]:
        self._load_encoder()
        assert self._encoder is not None

        embeddings = self._encoder.encode(documents, show_progress_bar=True)
        clusters = max(1, min(self.settings.target_topics, len(documents)))
        kmeans = KMeans(n_clusters=clusters, n_init="auto")
        topic_ids = kmeans.fit_predict(embeddings)

        assignments: Dict[int, List[str]] = {}
        for topic_id, text in zip(topic_ids, documents):
            assignments.setdefault(topic_id, []).append(text)

        results: List[TopicAssignment] = []
        for topic_id, texts in assignments.items():
            centroid = kmeans.cluster_centers_[topic_id]
            scores = self._encoder.encode(texts, convert_to_numpy=True)
            denom = (np.linalg.norm(scores, axis=1) * np.linalg.norm(centroid) + 1e-9)
            similarities = (scores @ centroid) / denom
            topic_idx = int(topic_id)
            label = self._labels.get(topic_idx, f"Topic {topic_idx}")
            top_indices = np.argsort(similarities)[::-1][:5]
            results.append(
                TopicAssignment(
                    topic_id=topic_idx,
                    label=label,
                    score=float(similarities[top_indices[0]]),
                    sentences=[texts[idx] for idx in top_indices],
                )
            )
        return results

    def update_labels(self, labels: Dict[int, str]) -> None:
        self._labels.update(labels)
