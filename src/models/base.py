"""Model-agnostic seam.

``GenerativeModel`` documents the interface a plugin's underlying flow model
exposes. ``BaseGenerativeModule`` is the shared LightningModule base that every
plugin's training module inherits.  Both are intentionally thin: plugin modules
keep their own training logic, optimizer construction and sampling code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import lightning as pl


@runtime_checkable
class GenerativeModel(Protocol):
    """Minimal conditional generative-model interface."""

    def sample(self, num_pts: int, condition):
        """Draw ``num_pts`` samples per condition."""
        ...

    def log_prob(self, x, condition):
        """Conditional log-density of ``x`` given ``condition``."""
        ...


class BaseGenerativeModule(pl.LightningModule):
    """Thin shared base for plugin LightningModules."""

    plugin_name: str = "base"
