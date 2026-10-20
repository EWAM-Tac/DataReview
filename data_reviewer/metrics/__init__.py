from .annotation import score_annotation
from .completeness import score_completeness
from .consistency import score_consistency
from .dynamics import score_dynamics
from .integrity import score_integrity
from .tactile import score_tactile
from .trainability import score_trainability

SCORERS = [
    score_completeness,
    score_integrity,
    score_dynamics,
    score_consistency,
    score_annotation,
    score_tactile,
    score_trainability,
]

__all__ = ["SCORERS"]
