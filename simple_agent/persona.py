"""Persona loading and system prompt construction from YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised by bare-pytest verify runs
    yaml = None


_REQUIRED_FIELDS = {"name", "identity", "voice", "expertise"}


def _parse_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_block(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _safe_load_yaml(text: str):
    if yaml is not None:
        return yaml.safe_load(text)

    stripped = text.strip()
    if ":" not in stripped:
        return _parse_scalar(stripped)

    lines = text.splitlines()
    idx = 0
    data: dict[str, object] = {}

    while idx < len(lines):
        line = lines[idx]
        if not line.strip() or line.lstrip().startswith("#"):
            idx += 1
            continue

        if line.startswith("  "):
            raise ValueError(f"Unexpected indentation: {line}")

        key, sep, raw_value = line.partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML line: {line}")

        value = raw_value.lstrip()
        if value in {">", "|"}:
            idx += 1
            block_lines: list[str] = []
            while idx < len(lines):
                candidate = lines[idx]
                if candidate.startswith("  "):
                    block_lines.append(candidate[2:])
                    idx += 1
                    continue
                if not candidate.strip():
                    block_lines.append("")
                    idx += 1
                    continue
                break
            data[key] = _parse_block(block_lines)
            continue

        if value == "":
            idx += 1
            items: list[str] = []
            while idx < len(lines):
                candidate = lines[idx]
                if not candidate.strip():
                    idx += 1
                    continue
                if not candidate.startswith("  - "):
                    break

                item_value = candidate[4:]
                if item_value in {">", "|"}:
                    idx += 1
                    block_lines: list[str] = []
                    while idx < len(lines):
                        block_candidate = lines[idx]
                        if block_candidate.startswith("    "):
                            block_lines.append(block_candidate[4:])
                            idx += 1
                            continue
                        if not block_candidate.strip():
                            block_lines.append("")
                            idx += 1
                            continue
                        break
                    items.append(_parse_block(block_lines))
                    continue

                items.append(_parse_scalar(item_value))
                idx += 1

            data[key] = items
            continue

        data[key] = _parse_scalar(value)
        idx += 1

    return data


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
        data = _safe_load_yaml(f.read())

    if not isinstance(data, dict):
        raise ValueError(f"Persona file must be a YAML mapping: {path}")

    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Persona {path.name} missing required fields: {missing}")

    return Persona(
        name=data["name"],
        identity=data["identity"],
        voice=data["voice"],
        expertise=data["expertise"],
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
