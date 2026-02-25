"""Persona loading and system prompt construction.

Personas are YAML configs with structured fields. The library loads them
and builds system prompts that internalize the persona as identity -
not "write like an analyst" but "you ARE an analyst."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_REQUIRED_FIELDS = {"name", "identity", "voice", "expertise"}


@dataclass
class Persona:
    """A persona that the agent can adopt for content generation."""

    name: str
    identity: str
    voice: str
    expertise: list[str]
    constraints: list[str] = field(default_factory=list)
    example_outputs: list[str] = field(default_factory=list)


def load_persona(path: Path | str) -> Persona:
    """Load a persona from a YAML file.

    Raises ValueError if required fields are missing.
    """
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
    """Build a system prompt that internalizes the persona as identity.

    The key insight: the prompt frames the persona as who you ARE,
    not as instructions to follow. This produces more consistent voice
    than instruction-style prompts.
    """
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
    """Load all persona YAML files from a directory."""
    directory = Path(directory)
    personas = []
    for path in sorted(directory.glob("*.yaml")):
        personas.append(load_persona(path))
    return personas
