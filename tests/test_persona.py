"""Tests for persona loading and system prompt construction."""

from pathlib import Path

import pytest
import yaml

from simple_agent.persona import Persona, build_system_prompt, list_personas, load_persona


def _write_persona_yaml(dir_path: Path, filename: str, data: dict) -> Path:
    path = dir_path / filename
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path


VALID_PERSONA = {
    "name": "Test Analyst",
    "identity": "A seasoned analyst with 10 years of experience.",
    "voice": "Direct and data-driven.",
    "expertise": ["Financial modeling", "Risk assessment"],
    "constraints": ["No em dashes", "No filler phrases"],
    "example_outputs": ["The yield looks strong at 5.2%."],
}


class TestLoadPersona:
    def test_loads_valid_yaml(self, tmp_path):
        path = _write_persona_yaml(tmp_path, "analyst.yaml", VALID_PERSONA)
        persona = load_persona(path)
        assert persona.name == "Test Analyst"
        assert len(persona.expertise) == 2
        assert len(persona.example_outputs) == 1

    def test_raises_on_missing_fields(self, tmp_path):
        incomplete = {"name": "Test", "voice": "Direct"}
        path = _write_persona_yaml(tmp_path, "bad.yaml", incomplete)
        with pytest.raises(ValueError, match="missing required fields"):
            load_persona(path)

    def test_raises_on_non_mapping(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_persona(path)

    def test_optional_fields_default(self, tmp_path):
        minimal = {
            "name": "Minimal",
            "identity": "A minimal persona.",
            "voice": "Terse.",
            "expertise": ["Testing"],
        }
        path = _write_persona_yaml(tmp_path, "minimal.yaml", minimal)
        persona = load_persona(path)
        assert persona.constraints == []
        assert persona.example_outputs == []


class TestBuildSystemPrompt:
    def test_includes_identity(self):
        persona = Persona(**VALID_PERSONA)
        prompt = build_system_prompt(persona)
        assert "Test Analyst" in prompt
        assert "seasoned analyst" in prompt

    def test_includes_voice(self):
        persona = Persona(**VALID_PERSONA)
        prompt = build_system_prompt(persona)
        assert "Direct and data-driven" in prompt

    def test_includes_expertise(self):
        persona = Persona(**VALID_PERSONA)
        prompt = build_system_prompt(persona)
        assert "Financial modeling" in prompt
        assert "Risk assessment" in prompt

    def test_includes_constraints(self):
        persona = Persona(**VALID_PERSONA)
        prompt = build_system_prompt(persona)
        assert "No em dashes" in prompt

    def test_includes_examples(self):
        persona = Persona(**VALID_PERSONA)
        prompt = build_system_prompt(persona)
        assert "5.2%" in prompt


class TestListPersonas:
    def test_loads_all_yaml_files(self, tmp_path):
        _write_persona_yaml(tmp_path, "a.yaml", VALID_PERSONA)
        modified = {**VALID_PERSONA, "name": "Second Persona"}
        _write_persona_yaml(tmp_path, "b.yaml", modified)
        personas = list_personas(tmp_path)
        assert len(personas) == 2
        names = [p.name for p in personas]
        assert "Test Analyst" in names
        assert "Second Persona" in names

    def test_empty_directory(self, tmp_path):
        personas = list_personas(tmp_path)
        assert personas == []
