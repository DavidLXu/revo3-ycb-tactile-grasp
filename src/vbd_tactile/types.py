"""Data contracts shared by the Newton adapter and taxel projector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactBatch:
    """Contact samples in world coordinates.

    ``position`` is in metres, ``force`` is in newtons and ``normal`` is a
    unit vector pointing out of the sensing surface.  Each row is one Newton
    contact.  Empty batches are valid and use arrays with zero rows.
    """

    position: np.ndarray
    force: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float32)
        force = np.asarray(self.force, dtype=np.float32)
        normal = np.asarray(self.normal, dtype=np.float32)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position must have shape (N, 3)")
        if force.shape != position.shape or normal.shape != position.shape:
            raise ValueError("force and normal must have shape (N, 3)")
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(force)) and np.all(np.isfinite(normal))):
            raise ValueError("contact position, force, and normal must be finite")
        lengths = np.linalg.norm(normal, axis=1)
        if np.any(lengths < 1.0e-8):
            raise ValueError("contact normals must be non-zero")
        if len(normal):
            normal = normal / lengths[:, None]
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "force", force)
        object.__setattr__(self, "normal", normal)

    @classmethod
    def empty(cls) -> "ContactBatch":
        empty = np.empty((0, 3), dtype=np.float32)
        return cls(empty, empty.copy(), empty.copy())


@dataclass(frozen=True)
class TactileFrame:
    """Output of one tactile update.

    ``pressure`` is in pascals, ``shear`` is in newtons per taxel, and
    ``normal_force`` is the normal load in newtons per taxel.  Taxel indexing
    is ``[row, column]`` with row increasing along layout ``v``.
    """

    pressure: np.ndarray
    shear: np.ndarray
    normal_force: np.ndarray
    total_force: np.ndarray
    center_of_pressure: np.ndarray
