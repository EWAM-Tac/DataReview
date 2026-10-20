"""DataReviewer — schema-agnostic robotics dataset quality scoring."""

from .aggregate import review_catalog, review_dataset
from .catalog import load_catalog
from .types import DimensionScore, ReviewReport

__all__ = [
    "review_dataset",
    "review_catalog",
    "load_catalog",
    "DimensionScore",
    "ReviewReport",
]
__version__ = "0.2.0"
