"""Generate auditable Git/PR artifact proposals without mutating git."""

from .proposal import propose_git_artifacts

__all__ = ["propose_git_artifacts"]
