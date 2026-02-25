"""Persona loading and system prompt construction from YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_REQUIRED_FIELDS = {"name", "identity", "voice", "expertise"}


@dataclass
class Persona:
    """A persona the agent can adopt for generation."""

    name: str
    identity: str
    voice: str
    expertise: list[str]
    constraints: list[str] = field(default_factory=list)
    example_outputs: list[str] = field(default_factory=list)


def load_persona(path: Path | str) -> Persona:
    """Load a persona from YAML. Raises ValueError on missing fields."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Persona file must be a YAML mapping: {path}")

    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Persona {path.name} missing required fields: {missing}")

    return Persona(
        name=data["name"],
        identity=data["identity"],
        voice=data["voice"],
        expertise=data.get("expertise", []),
        constraints=data.get("constraints", []),
        example_outputs=data.get("example_outputs", []),
    )


def build_system_prompt(persona: Persona) -> str:
    """Build a system prompt that frames the persona as identity, not instruction."""
    parts = [
        f"You are {persona.name}. {persona.identity}",
        f"\nYour voice: {persona.voice}",
        "\nYour expertise:",
    ]
    for area in persona.expertise:
        parts.append(f"- {area}")

    if persona.constraints:
        parts.append("\nRules you always follow:")
        for rule in persona.constraints:
            parts.append(f"- {rule}")

    if persona.example_outputs:
        parts.append("\nExamples of how you write:")
        for example in persona.example_outputs:
            parts.append(f'"""\n{example}\n"""')

    return "\n".join(parts)


def list_personas(directory: Path | str) -> list[Persona]:
    """Load all .yaml persona files from a directory."""
    directory = Path(directory)
    return [load_persona(p) for p in sorted(directory.glob("*.yaml"))]
