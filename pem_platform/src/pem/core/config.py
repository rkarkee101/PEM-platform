from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field, model_validator


class FactorType(str, Enum):
    continuous = "continuous"
    discrete = "discrete"
    categorical = "categorical"


class ResponseGoal(str, Enum):
    minimize = "minimize"
    maximize = "maximize"
    target = "target"


class FactorConfig(BaseModel):
    name: str
    type: FactorType
    description: str = ""
    units: str = ""

    # Continuous/discrete numeric bounds
    bounds: Optional[Tuple[float, float]] = None

    # Discrete numeric levels OR categorical string levels
    levels: Optional[List[Any]] = None

    # Whether this factor is allowed in DoE design generation.
    include_in_design: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "FactorConfig":
        if self.type in (FactorType.continuous, FactorType.discrete):
            if self.bounds is None and (self.levels is None or len(self.levels) < 2):
                raise ValueError(
                    f"Factor '{self.name}' ({self.type}) requires either bounds or >=2 levels."
                )
        if self.type == FactorType.categorical:
            if self.levels is None or len(self.levels) < 2:
                raise ValueError(f"Categorical factor '{self.name}' requires >=2 levels.")
        return self

    def low_high(self) -> Tuple[float, float]:
        """Return (low, high) numeric levels for 2-level coding."""
        if self.type == FactorType.categorical:
            raise ValueError(f"Factor '{self.name}' is categorical and has no numeric low/high.")
        if self.bounds is not None:
            lo, hi = self.bounds
            return float(lo), float(hi)
        assert self.levels is not None
        lo, hi = float(min(self.levels)), float(max(self.levels))
        return lo, hi


class ResponseConfig(BaseModel):
    name: str
    goal: ResponseGoal = ResponseGoal.minimize
    target: Optional[float] = None
    units: str = ""
    description: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "ResponseConfig":
        if self.goal == ResponseGoal.target and self.target is None:
            raise ValueError(f"Response '{self.name}' goal=target requires a numeric target.")
        return self


class ScalarizationConfig(BaseModel):
    """How to convert multiple responses into a single scalar objective."""

    method: str = Field(
        default="weighted_sum",
        description="Currently supported: weighted_sum",
    )
    weights: Dict[str, float] = Field(default_factory=dict)


class ConstraintPluginConfig(BaseModel):
    """Optional plugin to enforce feasibility constraints.

    The plugin must expose a function with signature:
        is_feasible(row: dict[str, Any]) -> bool

    where row contains factor values (natural units).
    """

    module: str
    function: str = "is_feasible"


class ProjectConfig(BaseModel):
    """Top-level config that makes the pipeline 'universal'."""

    project_name: str = "demo"

    factors: List[FactorConfig]
    responses: List[ResponseConfig]

    # Optional columns in the dataset that can be present but are not factors/responses.
    metadata_columns: List[str] = Field(default_factory=list)

    scalarization: ScalarizationConfig = Field(default_factory=ScalarizationConfig)

    constraint_plugin: Optional[ConstraintPluginConfig] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def factor_names(self, include_categorical: bool = True) -> List[str]:
        if include_categorical:
            return [f.name for f in self.factors]
        return [f.name for f in self.factors if f.type != FactorType.categorical]

    def response_names(self) -> List[str]:
        return [r.name for r in self.responses]

    def get_factor(self, name: str) -> FactorConfig:
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(f"Unknown factor '{name}'.")

    def get_response(self, name: str) -> ResponseConfig:
        for r in self.responses:
            if r.name == name:
                return r
        raise KeyError(f"Unknown response '{name}'.")

    def load_constraint_function(self) -> Optional[Callable[[Dict[str, Any]], bool]]:
        if self.constraint_plugin is None:
            return None
        import importlib

        mod = importlib.import_module(self.constraint_plugin.module)
        fn = getattr(mod, self.constraint_plugin.function)
        if not callable(fn):
            raise TypeError(
                f"Constraint plugin {self.constraint_plugin.module}:{self.constraint_plugin.function} is not callable."
            )
        return fn


