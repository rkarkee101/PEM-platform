from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from pem.core.config import FactorType, ProjectConfig

# Optional dependency: pyDOE2 provides robust implementations.
try:
    from pyDOE2 import ff2n, fracfact, lhs, pbdesign  # type: ignore
except Exception:  # pragma: no cover
    def ff2n(n: int) -> np.ndarray:
        """Fallback full factorial matrix in coded -1/+1."""

        if n <= 0:
            return np.empty((0, 0))
        mat = np.array(list(product([-1, 1], repeat=n)), dtype=float)
        return mat

    def fracfact(generator: str) -> np.ndarray:
        """Fallback fractional factorial generator.

        Supports generator tokens like:
            "a b c ab"  or  "a b c -ab"  or  "a b c abc"

        The run size is 2^(# base letters). Each token becomes one column.
        """

        tokens = [t.strip() for t in generator.split() if t.strip()]
        if not tokens:
            raise ValueError("Empty generator")

        # Base letters = unique single-letter tokens OR letters used in products.
        letters = sorted({ch for t in tokens for ch in t if ch.isalpha()})
        if not letters:
            raise ValueError("No factor letters found in generator")

        base = ff2n(len(letters))
        base_map = {letters[i]: base[:, i] for i in range(len(letters))}

        cols = []
        for tok in tokens:
            sign = 1.0
            t = tok
            if t.startswith("-"):
                sign = -1.0
                t = t[1:]
            # multiply the involved base columns
            col = np.ones(base.shape[0], dtype=float)
            for ch in t:
                if ch.isalpha():
                    col = col * base_map[ch]
            cols.append(sign * col)

        return np.column_stack(cols).astype(float)

    def lhs(n: int, samples: int, random_state=None) -> np.ndarray:
        """Fallback Latin hypercube in [0,1]."""

        rng = np.random.default_rng(random_state)
        cut = np.linspace(0, 1, samples + 1)
        u = rng.random((samples, n))
        a = cut[:samples]
        b = cut[1:samples + 1]
        rdpoints = u * (b - a)[:, None] + a[:, None]
        H = np.zeros_like(rdpoints)
        for j in range(n):
            order = rng.permutation(samples)
            H[:, j] = rdpoints[order, j]
        return H

    def pbdesign(n: int) -> np.ndarray:
        raise ImportError(
            "Plackett-Burman design requires pyDOE2. Install dependency pyDOE2>=1.3.0"
        )


@dataclass(frozen=True)
class DesignResult:
    coded: pd.DataFrame
    natural: pd.DataFrame


def _design_factors(cfg: ProjectConfig) -> List[str]:
    # Only non-categorical factors for classic 2-level designs.
    return [f.name for f in cfg.factors if f.include_in_design and f.type != FactorType.categorical]


def decode_coded_to_natural(cfg: ProjectConfig, coded_df: pd.DataFrame) -> pd.DataFrame:
    """Decode coded -1/+1 levels to natural units using each factor's bounds/levels."""

    natural = {}
    for name in coded_df.columns:
        f = cfg.get_factor(name)
        lo, hi = f.low_high()
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        natural[name] = mid + half * coded_df[name].astype(float)

        # If the factor is discrete with explicit levels, snap to nearest level for convenience.
        if f.type == FactorType.discrete and f.levels is not None:
            levels = np.array(sorted(map(float, f.levels)))
            vals = np.asarray(natural[name], dtype=float)
            snapped = levels[np.abs(vals[:, None] - levels[None, :]).argmin(axis=1)]
            natural[name] = snapped

    return pd.DataFrame(natural)


def full_factorial_2level(cfg: ProjectConfig) -> DesignResult:
    factors = _design_factors(cfg)
    if not factors:
        raise ValueError("No numeric factors available for design.")

    mat = ff2n(len(factors))  # -1/+1
    coded = pd.DataFrame(mat, columns=factors)
    natural = decode_coded_to_natural(cfg, coded)
    return DesignResult(coded=coded, natural=natural)


def fractional_factorial_2level(cfg: ProjectConfig, generator: str) -> DesignResult:
    """Generate a fractional factorial design using a generator string.

    Example generator: "a b c ab" (4 factors where the 4th is aliased as a*b)

    Notes:
    - When pyDOE2 is installed, this uses pyDOE2.fracfact.
    - The fallback implementation supports common generator patterns but is not exhaustive.
    """

    factors = _design_factors(cfg)
    if not factors:
        raise ValueError("No numeric factors available for design.")

    mat = fracfact(generator)  # -1/+1
    if mat.shape[1] != len(factors):
        raise ValueError(
            f"Generator produced {mat.shape[1]} columns but config has {len(factors)} design factors: {factors}"
        )

    coded = pd.DataFrame(mat, columns=factors)
    natural = decode_coded_to_natural(cfg, coded)
    return DesignResult(coded=coded, natural=natural)


def plackett_burman(cfg: ProjectConfig) -> DesignResult:
    """Generate a Plackett–Burman screening design."""

    factors = _design_factors(cfg)
    if not factors:
        raise ValueError("No numeric factors available for design.")

    mat = pbdesign(len(factors))
    coded = pd.DataFrame(mat, columns=factors)
    natural = decode_coded_to_natural(cfg, coded)
    return DesignResult(coded=coded, natural=natural)


def latin_hypercube(cfg: ProjectConfig, *, n_samples: int, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate LHS samples in natural units for numeric factors."""

    rng = np.random.default_rng(seed)
    factors = _design_factors(cfg)
    if not factors:
        raise ValueError("No numeric factors available for design.")

    unit = lhs(len(factors), samples=n_samples, random_state=rng)

    out = {}
    for i, name in enumerate(factors):
        f = cfg.get_factor(name)
        lo, hi = f.low_high()
        out[name] = lo + (hi - lo) * unit[:, i]

        if f.type == FactorType.discrete and f.levels is not None:
            levels = np.array(sorted(map(float, f.levels)))
            vals = np.asarray(out[name], dtype=float)
            snapped = levels[np.abs(vals[:, None] - levels[None, :]).argmin(axis=1)]
            out[name] = snapped

    return pd.DataFrame(out)
