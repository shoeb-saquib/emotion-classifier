"""
Structured config for BERTopic clustering variations.

Define each variation as a BERTopicConfig; name and model are built from
the config so they stay in sync.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import umap
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from bertopic.representation import LlamaCPP
from llama_cpp import Llama


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR
while _PROJECT_ROOT != _PROJECT_ROOT.parent and (_PROJECT_ROOT / "src").is_dir() is False:
    _PROJECT_ROOT = _PROJECT_ROOT.parent
LLM_MODELS_DIR = _PROJECT_ROOT / "llm_models"


@dataclass(frozen=True)
class BERTopicConfig:
    """Config for one BERTopic variation. None = use library default."""

    # Optional short label; if set, included in generated name
    label: Optional[str] = None

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # UMAP (None = UMAP default)
    umap_n_neighbors: Optional[int] = None
    umap_min_dist: Optional[float] = None
    umap_n_components: Optional[int] = None
    umap_metric: str = "cosine"

    # Clusterer
    clusterer: str = "kmeans"  # "kmeans" | "hdbscan"
    n_clusters: Optional[int] = 7
    hdbscan_min_cluster_size: Optional[int] = None

    # c-TF-IDF
    ctfidf_bm25_weighting: bool = False
    ctfidf_reduce_frequent_words: bool = False

    # BERTopic
    n_gram_range: Optional[Tuple[int, int]] = None
    min_topic_size: Optional[int] = None

    def build(self, use_llm: bool = True) -> BERTopic:
        """Build BERTopic instance from this config."""
        umap_model = self._build_umap()
        hdbscan_model = self._build_clusterer()
        ctfidf_model = self._build_ctfidf()
        kwargs = dict(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True,
        )
        if use_llm:
            kwargs["representation_model"] = self._build_llm()
        if ctfidf_model is not None:
            kwargs["ctfidf_model"] = ctfidf_model
        if self.n_gram_range is not None:
            kwargs["n_gram_range"] = self.n_gram_range
        if self.min_topic_size is not None:
            kwargs["min_topic_size"] = self.min_topic_size
        return BERTopic(**kwargs)

    def _build_umap(self) -> umap.UMAP:
        kwargs = {"n_jobs": 1}
        # Only override metric when we customize UMAP (so "default" stays true default)
        if any(
            x is not None
            for x in (
                self.umap_n_neighbors,
                self.umap_min_dist,
                self.umap_n_components,
            )
        ):
            kwargs["metric"] = self.umap_metric
        if self.umap_n_neighbors is not None:
            kwargs["n_neighbors"] = self.umap_n_neighbors
        if self.umap_min_dist is not None:
            kwargs["min_dist"] = self.umap_min_dist
        if self.umap_n_components is not None:
            kwargs["n_components"] = self.umap_n_components
        return umap.UMAP(**kwargs)

    def _build_clusterer(self):
        if self.clusterer == "kmeans":
            n = self.n_clusters if self.n_clusters is not None else 7
            return KMeans(n_clusters=n)
        if self.clusterer == "hdbscan":
            kwargs = {}
            if self.hdbscan_min_cluster_size is not None:
                kwargs["min_cluster_size"] = self.hdbscan_min_cluster_size
            return HDBSCAN(**kwargs)
        raise ValueError(f"Unknown clusterer: {self.clusterer}")

    def _build_ctfidf(self) -> Optional[ClassTfidfTransformer]:
        if not self.ctfidf_bm25_weighting and not self.ctfidf_reduce_frequent_words:
            return None
        return ClassTfidfTransformer(
            bm25_weighting=self.ctfidf_bm25_weighting,
            reduce_frequent_words=self.ctfidf_reduce_frequent_words,
        )

    def _build_llm(self) -> LlamaCPP:
        """Build LlamaCPP-based representation model pointing to llm_models/."""
        model_path = LLM_MODELS_DIR / "google_gemma-3-4b-it-Q5_K_S.gguf"
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=2048,
            stop=["Q:", "\n"],
        )
        return LlamaCPP(llm)

    def name(self) -> str:
        """Generate a readable variation name from the config."""
        parts = []

        # UMAP
        u = []
        if self.umap_n_neighbors is not None:
            u.append(f"n_neighbors={self.umap_n_neighbors}")
        if self.umap_min_dist is not None:
            u.append(f"min_dist={self.umap_min_dist}")
        if self.umap_n_components is not None:
            u.append(f"n_components={self.umap_n_components}")
        if u:
            u.append(f"metric={self.umap_metric!r}")
        parts.append("UMAP(" + ", ".join(u) + ")" if u else "UMAP(default)")

        # Clusterer
        if self.clusterer == "kmeans":
            n = self.n_clusters if self.n_clusters is not None else 7
            parts.append(f"KMeans({n})")
        else:
            if self.hdbscan_min_cluster_size is not None:
                parts.append(f"HDBSCAN(min_cluster_size={self.hdbscan_min_cluster_size})")
            else:
                parts.append("HDBSCAN(default)")

        # c-TF-IDF
        ctf = []
        if self.ctfidf_bm25_weighting:
            ctf.append("bm25_weighting=True")
        if self.ctfidf_reduce_frequent_words:
            ctf.append("reduce_frequent_words=True")
        if ctf:
            parts.append("c-TF-IDF(" + ", ".join(ctf) + ")")

        # BERTopic options
        if self.n_gram_range is not None:
            parts.append(f"n_gram_range={self.n_gram_range}")
        if self.min_topic_size is not None:
            parts.append(f"min_topic_size={self.min_topic_size}")

        if self.label:
            parts.insert(0, self.label + ":")

        return " + ".join(parts)

    def code_name(self) -> str:
        """
        Generate a unique, filesystem-safe abbreviated name for this config.
        Same config always yields the same string. Safe for use as a filename.
        """
        parts = []

        # Clusterer: k = kmeans, h = hdbscan
        if self.clusterer == "kmeans":
            n = self.n_clusters if self.n_clusters is not None else 7
            parts.append(f"k_{n}")
        else:
            if self.hdbscan_min_cluster_size is not None:
                parts.append(f"h_m{self.hdbscan_min_cluster_size}")
            else:
                parts.append("h_default")

        # UMAP overrides (only if any set)
        if any(
            x is not None
            for x in (
                self.umap_n_neighbors,
                self.umap_min_dist,
                self.umap_n_components,
            )
        ):
            u = ["u"]
            if self.umap_n_neighbors is not None:
                u.append(f"n{self.umap_n_neighbors}")
            if self.umap_min_dist is not None:
                md_val = int(round(self.umap_min_dist * 100))
                u.append(f"md{md_val}")
            if self.umap_n_components is not None:
                u.append(f"c{self.umap_n_components}")
            parts.append("_".join(u))

        # c-TF-IDF
        if self.ctfidf_bm25_weighting or self.ctfidf_reduce_frequent_words:
            ctf = ["ctfidf"]
            if self.ctfidf_bm25_weighting:
                ctf.append("b")
            if self.ctfidf_reduce_frequent_words:
                ctf.append("r")
            parts.append("_".join(ctf))

        # BERTopic
        if self.n_gram_range is not None:
            parts.append(f"ng{self.n_gram_range[0]}_{self.n_gram_range[1]}")
        if self.min_topic_size is not None:
            parts.append(f"mts{self.min_topic_size}")

        if self.label:
            # Sanitize label for filename: lowercase, replace spaces/special with _
            safe_label = "".join(
                c if c.isalnum() or c == "_" else "_" for c in self.label.lower()
            ).strip("_")
            if safe_label:
                parts.insert(0, safe_label)

        return "_".join(parts)


# Variations to run when executing the BERTopic comparison pipeline.
# Add or remove configs here to change which variations are evaluated.
BERTOPIC_VARIATIONS: Tuple[BERTopicConfig, ...] = (
    BERTopicConfig(clusterer="hdbscan"),
    BERTopicConfig(clusterer="kmeans", n_clusters=7),
    BERTopicConfig(clusterer="hdbscan", hdbscan_min_cluster_size=20),
    # BERTopicConfig(
    #     clusterer="kmeans",
    #     n_clusters=20,
    #     umap_n_neighbors=10,
    #     umap_min_dist=0.0,
    # ),
    # BERTopicConfig(
    #     clusterer="kmeans",
    #     n_clusters=7,
    #     umap_n_neighbors=15,
    #     umap_n_components=5,
    #     umap_min_dist=0.0,
    # ),
    # BERTopicConfig(
    #     clusterer="kmeans",
    #     n_clusters=7,
    #     umap_n_neighbors=30,
    #     umap_n_components=5,
    #     umap_min_dist=0.0,
    # ),
)
