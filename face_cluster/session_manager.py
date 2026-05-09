"""Session management for iterative face clustering.

A Session represents one clustering experiment on an album:
  - base/: the expensive one-time output (detect, embed, quality gate)
  - chain_NN/: zero or more clustering chains, each a sequence of steps

A Step is a materialized clustering state (a folder with clusters.csv etc).
Steps are created only when clustering actually runs (cluster, recluster, merge, remerge).
User labels accumulate in chain_NN/pending_labels.json without creating steps.

session.json schema (schema_version=1):
{
  "schema_version": 1,
  "session_id": "<album_name>_<YYYYMMDD_HHMMSS>",
  "created_at": "<ISO8601>",
  "source_album": "<str>",
  "base": {
    "folder": "base",
    "timestamp": "<ISO8601>",
    "params_summary": "<str>",
    "n_faces": <int>,
    "n_images": <int>
  },
  "chains": [
    {
      "chain_id": "chain_01",
      "folder": "chain_01",
      "created_at": "<ISO8601>",
      "branched_from": null | {"chain_id": "<str>", "step_index": <int>},
      "has_pending_labels": <bool>,
      "steps": [
        {
          "step_index": 0,
          "folder": "step_00_cluster",
          "type": "cluster" | "recluster" | "merge" | "remerge" | "split",
          "timestamp": "<ISO8601>",
          "params": {},
          "result_summary": "<str>"
        }
      ]
    }
  ],
  "active_chain": "<chain_id>" | null
}
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SESSION_FILE = "session.json"
_PENDING_LABELS_FILE = "pending_labels.json"
_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Step:
    step_index: int
    folder: str       # relative to chain folder
    type: str         # "cluster" | "recluster" | "merge" | "remerge" | "split"
    timestamp: str
    params: dict
    result_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "folder": self.folder,
            "type": self.type,
            "timestamp": self.timestamp,
            "params": self.params,
            "result_summary": self.result_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            step_index=d["step_index"],
            folder=d["folder"],
            type=d["type"],
            timestamp=d["timestamp"],
            params=d.get("params", {}),
            result_summary=d.get("result_summary", ""),
        )


@dataclass
class Chain:
    chain_id: str
    folder: str       # relative to session root
    created_at: str
    branched_from: Optional[dict]    # {"chain_id": str, "step_index": int} | None
    has_pending_labels: bool
    steps: list[Step]

    def to_dict(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "folder": self.folder,
            "created_at": self.created_at,
            "branched_from": self.branched_from,
            "has_pending_labels": self.has_pending_labels,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chain":
        return cls(
            chain_id=d["chain_id"],
            folder=d["folder"],
            created_at=d["created_at"],
            branched_from=d.get("branched_from"),
            has_pending_labels=d.get("has_pending_labels", False),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
        )


@dataclass
class Session:
    session_id: str
    created_at: str
    source_album: str
    session_root: Path
    base: dict         # see schema above
    chains: list[Chain]
    active_chain: Optional[str]

    def to_dict(self) -> dict:
        return {
            "schema_version": _SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "source_album": self.source_album,
            "base": self.base,
            "chains": [c.to_dict() for c in self.chains],
            "active_chain": self.active_chain,
        }

    @classmethod
    def from_dict(cls, d: dict, session_root: Path) -> "Session":
        return cls(
            session_id=d["session_id"],
            created_at=d["created_at"],
            source_album=d.get("source_album", ""),
            session_root=session_root,
            base=d.get("base", {}),
            chains=[Chain.from_dict(c) for c in d.get("chains", [])],
            active_chain=d.get("active_chain"),
        )


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages session.json and pending_labels.json on disk.

    All mutating methods call _save() immediately to persist state.
    No UI or Streamlit code here — pure data/disk logic.
    """

    # ------------------------------------------------------------------ create / load

    def create(
        self,
        session_root: Path,
        source_album: str,
        base_summary: dict,
    ) -> Session:
        """Create a new session directory and session.json.

        Args:
            session_root: Root directory for the session (created if absent).
            source_album: Path to the source album (for display only).
            base_summary: Dict with optional keys: params_summary, n_faces, n_images.

        Returns:
            The new Session object.
        """
        session_root.mkdir(parents=True, exist_ok=True)
        base_folder = session_root / "base"
        base_folder.mkdir(exist_ok=True)

        now = _now_iso()
        album_name = Path(source_album).name if source_album else "session"
        ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{album_name}_{ts_compact}"

        base_meta = {
            "folder": "base",
            "timestamp": now,
            "params_summary": base_summary.get("params_summary", ""),
            "n_faces": base_summary.get("n_faces", 0),
            "n_images": base_summary.get("n_images", 0),
        }

        session = Session(
            session_id=session_id,
            created_at=now,
            source_album=str(source_album),
            session_root=session_root,
            base=base_meta,
            chains=[],
            active_chain=None,
        )
        self._save(session)
        logger.debug("Session created: %s at %s", session_id, session_root)
        return session

    def load(self, session_root: Path) -> Optional[Session]:
        """Load an existing session from session.json.

        Returns None if session.json does not exist or is malformed.
        """
        path = session_root / _SESSION_FILE
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            session = Session.from_dict(data, session_root)
            logger.debug("Session loaded: %s", session.session_id)
            return session
        except Exception as exc:
            logger.warning("Failed to load session from %s: %s", path, exc)
            return None

    def update_base_summary(self, session: Session, n_faces: int, n_images: int) -> None:
        """Update n_faces / n_images in the base metadata after the pipeline run completes."""
        session.base["n_faces"] = n_faces
        session.base["n_images"] = n_images
        self._save(session)

    # ------------------------------------------------------------------ chains

    def create_chain(
        self,
        session: Session,
        branched_from: Optional[tuple[str, int]] = None,
    ) -> Chain:
        """Create a new chain in the session.

        Args:
            branched_from: (chain_id, step_index) to copy steps from, or None for a fresh chain.
                           Steps 0..step_index are copied as definitions (not re-executed here).

        Returns:
            The new Chain object (already appended to session.chains and persisted).
        """
        n = len(session.chains) + 1
        chain_id = f"chain_{n:02d}"
        folder = chain_id
        chain_dir = session.session_root / folder
        chain_dir.mkdir(exist_ok=True)

        branched_from_dict: Optional[dict] = None
        initial_steps: list[Step] = []

        if branched_from is not None:
            src_chain_id, src_step_index = branched_from
            src_chain = self.get_chain(session, src_chain_id)
            branched_from_dict = {"chain_id": src_chain_id, "step_index": src_step_index}
            # Copy step definitions (not output folders) up to and including src_step_index
            initial_steps = [
                Step(
                    step_index=s.step_index,
                    folder=s.folder,
                    type=s.type,
                    timestamp=s.timestamp,
                    params=dict(s.params),
                    result_summary=s.result_summary,
                )
                for s in src_chain.steps
                if s.step_index <= src_step_index
            ]

        chain = Chain(
            chain_id=chain_id,
            folder=folder,
            created_at=_now_iso(),
            branched_from=branched_from_dict,
            has_pending_labels=False,
            steps=initial_steps,
        )
        session.chains.append(chain)
        session.active_chain = chain_id
        self._save(session)
        logger.debug("Chain created: %s (branched_from=%s)", chain_id, branched_from_dict)
        return chain

    # ------------------------------------------------------------------ steps

    def append_step(
        self,
        session: Session,
        chain_id: str,
        step_type: str,
        params: dict,
    ) -> tuple[Step, Path]:
        """Append a new step to the chain and create its output folder.

        Returns:
            (Step, absolute_output_path) — the folder is created and ready to write into.
        """
        chain = self.get_chain(session, chain_id)
        step_index = len(chain.steps)
        folder = f"step_{step_index:02d}_{step_type}"
        step_dir = session.session_root / chain.folder / folder
        step_dir.mkdir(parents=True, exist_ok=True)

        step = Step(
            step_index=step_index,
            folder=folder,
            type=step_type,
            timestamp=_now_iso(),
            params=params,
            result_summary="",
        )
        chain.steps.append(step)
        self._save(session)
        logger.debug("Step appended: %s/%s -> %s", chain_id, folder, step_dir)
        return step, step_dir

    def update_step_result(
        self,
        session: Session,
        chain_id: str,
        step_index: int,
        result_summary: str,
    ) -> None:
        """Update the result summary string for a step after it has executed."""
        chain = self.get_chain(session, chain_id)
        for s in chain.steps:
            if s.step_index == step_index:
                s.result_summary = result_summary
                break
        self._save(session)

    # ------------------------------------------------------------------ pending labels

    def save_pending_labels(
        self,
        session: Session,
        chain_id: str,
        labels: list[dict],
    ) -> None:
        """Append labels to chain_NN/pending_labels.json (accumulating without remerge).

        Each label dict should contain at minimum: pair (list[str]) and decision (str).
        Existing pending labels are preserved and extended.
        """
        chain = self.get_chain(session, chain_id)
        pending_path = session.session_root / chain.folder / _PENDING_LABELS_FILE
        existing = self._read_pending_labels(pending_path)
        existing.extend(labels)
        _write_json_atomic(pending_path, {"labels": existing})
        chain.has_pending_labels = len(existing) > 0
        self._save(session)
        logger.debug("Saved %d pending labels to %s (total: %d)", len(labels), chain_id, len(existing))

    def consume_pending_labels(
        self,
        session: Session,
        chain_id: str,
    ) -> list[dict]:
        """Read and clear pending_labels.json.

        Returns the consumed labels. The file is deleted after reading.
        Call this before creating a merge step to bundle all pending decisions into the step.
        """
        chain = self.get_chain(session, chain_id)
        pending_path = session.session_root / chain.folder / _PENDING_LABELS_FILE
        labels = self._read_pending_labels(pending_path)
        if pending_path.exists():
            pending_path.unlink()
        chain.has_pending_labels = False
        self._save(session)
        logger.debug("Consumed %d pending labels from %s", len(labels), chain_id)
        return labels

    def get_pending_labels(
        self,
        session: Session,
        chain_id: str,
    ) -> list[dict]:
        """Read pending labels without consuming them."""
        chain = self.get_chain(session, chain_id)
        pending_path = session.session_root / chain.folder / _PENDING_LABELS_FILE
        return self._read_pending_labels(pending_path)

    def has_pending_labels(self, session: Session, chain_id: str) -> bool:
        """Return True if there are any accumulated pending labels for the chain."""
        chain = self.get_chain(session, chain_id)
        pending_path = session.session_root / chain.folder / _PENDING_LABELS_FILE
        if not pending_path.exists():
            return False
        return len(self._read_pending_labels(pending_path)) > 0

    # ------------------------------------------------------------------ accessors

    def get_chain(self, session: Session, chain_id: str) -> Chain:
        for c in session.chains:
            if c.chain_id == chain_id:
                return c
        raise KeyError(f"Chain {chain_id!r} not found in session {session.session_id!r}")

    def get_active_chain(self, session: Session) -> Optional[Chain]:
        if session.active_chain is None:
            return None
        try:
            return self.get_chain(session, session.active_chain)
        except KeyError:
            return None

    def set_active_chain(self, session: Session, chain_id: str) -> None:
        self.get_chain(session, chain_id)  # validate existence
        session.active_chain = chain_id
        self._save(session)

    def get_step_output_dir(
        self,
        session: Session,
        chain_id: str,
        step_index: int,
    ) -> Path:
        """Return the absolute path to a step's output folder."""
        chain = self.get_chain(session, chain_id)
        for s in chain.steps:
            if s.step_index == step_index:
                return session.session_root / chain.folder / s.folder
        raise KeyError(f"Step {step_index} not found in chain {chain_id!r}")

    def get_base_dir(self, session: Session) -> Path:
        """Return the absolute path to the base/ output folder."""
        return session.session_root / session.base.get("folder", "base")

    def get_latest_step_dir(self, session: Session, chain_id: str) -> Path:
        """Return the output dir of the last step in the chain.

        Falls back to base/ if the chain has no steps yet.
        """
        chain = self.get_chain(session, chain_id)
        if not chain.steps:
            return self.get_base_dir(session)
        last_step = max(chain.steps, key=lambda s: s.step_index)
        return session.session_root / chain.folder / last_step.folder

    # ------------------------------------------------------------------ internal

    def _save(self, session: Session) -> None:
        """Atomically overwrite session.json."""
        path = session.session_root / _SESSION_FILE
        _write_json_atomic(path, session.to_dict())

    def _read_pending_labels(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("labels", [])
        except Exception as exc:
            logger.warning("Failed to read pending labels from %s: %s", path, exc)
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON atomically: write to temp then rename (works on Windows within same volume)."""
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_session_json(path: Path) -> dict:
    """Load raw session.json as a dict. Used by contract tests."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
