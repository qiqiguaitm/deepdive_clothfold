import json
from pathlib import Path

import pytest

from pi05_r3_semantic_prompt import SemanticPromptFormatter


@pytest.fixture
def semantic_files(tmp_path: Path) -> tuple[Path, Path]:
    vocabulary = [
        {"task": "stack", "local_id": 0, "name": "place_red_base"},
        {"task": "stack", "local_id": 1, "name": "stack_green_on_red"},
    ]
    task_map = {"stack": 3}
    vocabulary_path = tmp_path / "vocabulary.json"
    task_map_path = tmp_path / "task_map.json"
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    task_map_path.write_text(json.dumps(task_map), encoding="utf-8")
    return vocabulary_path, task_map_path


def observation(current: int = 0, nxt: int = 1, available: bool = True) -> dict:
    return {
        "lmwm_transition_task": 3,
        "lmwm_transition_current": current,
        "lmwm_transition_next": nxt,
        "lmwm_transition_mask": available,
    }


def test_semantic_next_uses_future_event(semantic_files: tuple[Path, Path]) -> None:
    formatter = SemanticPromptFormatter(*semantic_files, mode="semantic-next")
    assert formatter.format("Stack the blocks.", observation()) == (
        "Stack the blocks. Next subtask: stack green on red."
    )
    assert formatter.format("Stack the blocks.", observation(current=1, nxt=2)).endswith(
        "Next subtask: complete the task."
    )


def test_controls_preserve_same_base_prompt(semantic_files: tuple[Path, Path]) -> None:
    base = "Stack the blocks."
    assert SemanticPromptFormatter(*semantic_files, mode="none").format(base, observation()) == base
    assert SemanticPromptFormatter(*semantic_files, mode="generic-stage").format(base, observation()).endswith(
        "Current stage ID: 0. Next stage ID: 1."
    )
    assert SemanticPromptFormatter(*semantic_files, mode="semantic-current").format(base, observation()).endswith(
        "Current subtask: place red base."
    )


def test_missing_or_unknown_metadata_fails_closed(semantic_files: tuple[Path, Path]) -> None:
    formatter = SemanticPromptFormatter(*semantic_files, mode="semantic-next")
    assert formatter.format("base", observation(available=False)) == "base"
    with pytest.raises(ValueError, match="unknown semantic task"):
        formatter.format("base", {**observation(), "lmwm_transition_task": 99})
