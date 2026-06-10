"""
Packs: the saved, exportable unit of user work.

A pack is a single JSON file bundling everything a user built — models
(runnable simulation specs), named scenarios (parameter variations on a
model), and metadata. Packs are 100% data: no code, ever. Anything that
creates or manages packs (a chat UI, a cloud service) lives outside the
library; anything that *runs* them needs only this module.

Format:

    {
      "format": "agentstan-pack",
      "schema_version": 1,
      "name": "goblin-economy",
      "description": "Economy balance tests for Goblin Keep",
      "metadata": {"author": "...", "created": "2026-06-10", "tags": []},
      "models": {
        "base": { <simulation spec: environment, agent_types, ...> }
      },
      "scenarios": {
        "gold-rush": {
          "model": "base",
          "description": "Double the gold faucet",
          "overrides": {"agent_types.miner.initial_count": 40},
          "steps": 500,
          "seed": 7
        }
      },
      "default_model": "base"
    }

Scenario overrides are dot-paths into the model spec, the same notation
used by experiment.batch_run(vary=...).

Usage:

    from agentstan.pack import load

    pack = load("goblin-economy.pack.json")
    results = pack.run("gold-rush")          # scenario by name
    results = pack.run(steps=200)            # default model, no overrides
    spec = pack.spec("gold-rush")            # resolved spec, if you want
                                             # to drive Simulation yourself
"""

import copy
import json
from typing import Any, Dict, List, Optional

from .core.simulation import Simulation

FORMAT_NAME = "agentstan-pack"
SCHEMA_VERSION = 1


class PackError(ValueError):
    """A pack file is malformed. Message says what and where."""


def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set a nested dict value in place by dot-path."""
    keys = path.split(".")
    target = d
    for key in keys[:-1]:
        if not isinstance(target, dict) or key not in target:
            raise PackError(f"override path '{path}' not found in model spec")
        target = target[key]
    if not isinstance(target, dict):
        raise PackError(f"override path '{path}' not found in model spec")
    target[keys[-1]] = value


class Pack:
    """An in-memory pack. Construct via load() / from_dict()."""

    def __init__(self, data: Dict[str, Any]):
        _validate_pack(data)
        self.data = data

    # --- Introspection ---

    @property
    def name(self) -> str:
        return self.data.get("name", "")

    @property
    def description(self) -> str:
        return self.data.get("description", "")

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.data.get("metadata", {})

    @property
    def models(self) -> List[str]:
        return list(self.data["models"].keys())

    @property
    def scenarios(self) -> List[str]:
        return list(self.data.get("scenarios", {}).keys())

    @property
    def default_model(self) -> str:
        return self.data.get("default_model") or self.models[0]

    # --- Resolution ---

    def spec(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve a runnable spec by scenario or model name.

        Looks up scenarios first, then models. None means the default model.
        Returns a deep copy — safe to mutate or hand to Simulation.
        """
        if name is None:
            name = self.default_model

        scenarios = self.data.get("scenarios", {})
        if name in scenarios:
            scenario = scenarios[name]
            spec = copy.deepcopy(self.data["models"][scenario["model"]])
            for path, value in scenario.get("overrides", {}).items():
                _set_nested(spec, path, value)
            if "seed" in scenario:
                spec["seed"] = scenario["seed"]
            if "steps" in scenario:
                spec["steps"] = scenario["steps"]
            return spec

        if name in self.data["models"]:
            return copy.deepcopy(self.data["models"][name])

        raise PackError(
            f"'{name}' is not a model or scenario in pack '{self.name}' — "
            f"models: {self.models}, scenarios: {self.scenarios}"
        )

    # --- Execution ---

    def run(self, name: Optional[str] = None, steps: Optional[int] = None,
            seed: Optional[int] = None, collectors: Optional[list] = None) -> Dict[str, Any]:
        """Run a scenario or model. Arguments override pack values.

        Collectors (DataCollector, FrameRecorder, Observer, ...) are
        attached to the simulation before it runs.
        """
        spec = self.spec(name)
        run_steps = steps or spec.pop("steps", 200)
        sim = Simulation(spec, seed=seed if seed is not None else spec.get("seed"))
        for collector in collectors or []:
            sim.add_collector(collector)
        return sim.run(run_steps)

    def validate(self, deep: bool = False) -> None:
        """
        Re-validate the pack. With deep=True, every model and scenario is
        resolved and constructed as a real Simulation (catches bad rules,
        bad environments — everything the engine would reject).
        """
        _validate_pack(self.data)
        if deep:
            for name in self.models + self.scenarios:
                spec = self.spec(name)
                spec.pop("steps", None)
                try:
                    Simulation(spec)
                except Exception as e:
                    raise PackError(f"'{name}' fails engine validation: {e}")

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.data, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pack":
        return cls(data)

    @classmethod
    def new(cls, name: str, model: Dict[str, Any],
            description: str = "", metadata: Optional[Dict] = None) -> "Pack":
        """Create a minimal single-model pack around a spec."""
        return cls({
            "format": FORMAT_NAME,
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "description": description,
            "metadata": metadata or {},
            "models": {"base": model},
            "default_model": "base",
        })

    def __repr__(self):
        return (f"Pack(name={self.name!r}, models={self.models}, "
                f"scenarios={self.scenarios})")


def _validate_pack(data: Any) -> None:
    """Structural validation with precise error messages."""
    if not isinstance(data, dict):
        raise PackError(f"pack must be a dict, got {type(data).__name__}")

    if data.get("format") != FORMAT_NAME:
        raise PackError(
            f'pack missing "format": "{FORMAT_NAME}" — '
            f"got {data.get('format')!r}"
        )

    version = data.get("schema_version")
    if not isinstance(version, int):
        raise PackError('pack missing integer "schema_version"')
    if version > SCHEMA_VERSION:
        raise PackError(
            f"pack schema_version {version} is newer than this library "
            f"supports ({SCHEMA_VERSION}) — upgrade agentstan"
        )

    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise PackError('pack needs a non-empty "models" dict')
    for name, spec in models.items():
        if not isinstance(spec, dict):
            raise PackError(f"models['{name}'] must be a spec dict")
        try:
            Simulation._validate_spec(spec)
        except ValueError as e:
            raise PackError(f"models['{name}'] is not a valid spec: {e}")

    scenarios = data.get("scenarios", {})
    if not isinstance(scenarios, dict):
        raise PackError('"scenarios" must be a dict')
    for name, scenario in scenarios.items():
        where = f"scenarios['{name}']"
        if not isinstance(scenario, dict):
            raise PackError(f"{where} must be a dict")
        model_ref = scenario.get("model")
        if model_ref not in models:
            raise PackError(
                f"{where} references model {model_ref!r} — "
                f"available models: {list(models)}"
            )
        if name in models:
            raise PackError(
                f"'{name}' is both a model and a scenario — names must be unique"
            )
        overrides = scenario.get("overrides", {})
        if not isinstance(overrides, dict):
            raise PackError(f"{where}.overrides must be a dict of dot-path: value")

    default = data.get("default_model")
    if default is not None and default not in models:
        raise PackError(
            f'default_model {default!r} not in models: {list(models)}'
        )


def load(path: str) -> Pack:
    """Load and validate a pack from a JSON file."""
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise PackError(f"{path} is not valid JSON: {e}")
    return Pack(data)


def save(pack: Pack, path: str) -> None:
    """Write a pack to a JSON file."""
    with open(path, "w") as f:
        f.write(pack.to_json())
        f.write("\n")
