"""Load taxonomy.yaml and render it for prompts.

Spec: SPEC.md §8, §7.9.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_taxonomy(path: Path = Path("taxonomy.yaml")) -> dict:
    """Read the failure taxonomy: {type_name: {definition, signature, causes, fixes, metric}}.

    A relative path that does not exist from the current directory is tried again
    from the repository root, so the UI and scripts work from anywhere.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def render_taxonomy(tax: dict) -> str:
    """Readable text block for the system prompt, one entry per failure type."""
    blocks = []
    for name, entry in tax.items():
        lines = [
            f"{name}",
            f"  definition: {entry.get('definition', '')}",
            f"  signature: {entry.get('signature', '')}",
            f"  typical causes: {'; '.join(_as_list(entry.get('causes')))}",
            f"  typical fixes: {'; '.join(_as_list(entry.get('fixes')))}",
            f"  metric to monitor: {entry.get('metric', '')}",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
