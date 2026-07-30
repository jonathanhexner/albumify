"""Persist named parameter profiles for the face clustering app.

Profiles are stored as JSON files under ~/.sim_bench/profiles/<name>.json.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from face_cluster import run_history_db
from face_cluster._paths import profiles_dir as _default_profiles_dir

logger = logging.getLogger(__name__)


@dataclass
class ProfileStore:
    profiles_dir: Path = field(default_factory=_default_profiles_dir)

    def _path(self, name: str) -> Path:
        return self.profiles_dir / f"{name}.json"

    def save(self, name: str, params: Dict) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(json.dumps(params, indent=2), encoding="utf-8")
        logger.debug("Saved profile '%s' (%d params)", name, len(params))
        _aid = run_history_db.start_action(
            "profile_save", payload={"profile_name": name, "n_params": len(params), "params": params}
        )
        run_history_db.complete_action(_aid)

    def load(self, name: str) -> Dict:
        p = self._path(name)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def list_names(self) -> List[str]:
        if not self.profiles_dir.exists():
            return []
        return sorted(p.stem for p in self.profiles_dir.glob("*.json"))
