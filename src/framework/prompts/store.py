"""Versioned prompts: store prompts in files (or DB) with version tags."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PromptVersion:
    """Single version of a prompt: name, version tag, body."""

    name: str
    version: str
    body: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PromptStore:
    """File-based versioned prompt store. Layout: base_path/{name}_{version}.txt or base_path/{name}/{version}.txt."""

    def __init__(self, base_path: str = "./data/prompts"):
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, name: str, version: str) -> Path:
        """Path for prompt file: base_path/name_version.txt."""
        safe_name = name.replace("/", "_").strip() or "default"
        safe_version = version.replace("/", "_").strip() or "v1"
        return self._base / f"{safe_name}_{safe_version}.txt"

    def get(self, name: str, version: str = "v1") -> Optional[PromptVersion]:
        """Load prompt by name and version tag. Returns None if not found."""
        path = self._path_for(name, version)
        if not path.exists():
            return None
        body = path.read_text(encoding="utf-8").strip()
        return PromptVersion(name=name, version=version, body=body, metadata={})

    def put(self, name: str, version: str, body: str, metadata: Optional[dict] = None) -> PromptVersion:
        """Save prompt with version tag."""
        path = self._path_for(name, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return PromptVersion(name=name, version=version, body=body, metadata=metadata or {})

    def list_versions(self, name: str) -> list[str]:
        """List version tags for a prompt name (files matching name_*.txt)."""
        safe_name = name.replace("/", "_").strip() or "default"
        prefix = f"{safe_name}_"
        versions = []
        for f in self._base.iterdir():
            if f.is_file() and f.suffix == ".txt" and f.stem.startswith(prefix):
                ver = f.stem[len(prefix):]
                if ver:
                    versions.append(ver)
        return sorted(versions)

    def list_names(self) -> list[str]:
        """List prompt names (unique prefix before _version)."""
        names = set()
        for f in self._base.iterdir():
            if f.is_file() and f.suffix == ".txt" and "_" in f.stem:
                # assume last _ is version
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2:
                    names.add(parts[0])
        return sorted(names)
