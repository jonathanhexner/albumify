"""Shared fakes for spec-112 control-plane tests: a finished PipelineContext stand-in.

Six photos: a, c picks (c shot before a); b near-dup of a; d not selected; q removed before scoring
(only a step decision says why); s set aside by triage. Three faces, two people.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

NAMES = ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "q.jpg", "s.jpg")


def _face(fid, path, person):
    return SimpleNamespace(face_id=fid, face_index=0, image_path=path, bbox=(0, 0, 10, 10), det_score=0.9,
                           blur_score=50.0, pose=(0.0, 0.0, 0.0), is_core=True,
                           rejection_reason=None, embedding_normalized=np.ones(4) * fid,
                           aligned_face=np.full((112, 112, 3), 128, np.uint8), _person=person)


def _context(images):
    """``images``: the paths the curator hands the pipeline (spec-112 A2: shared input store)."""
    given = {Path(p).name: str(p) for p in images}
    img = {n: given.get(n, f"/not-staged/{n}") for n in NAMES}
    t = lambda h: SimpleNamespace(timestamp=datetime(2026, 8, 22, h, 0), lat=48.0, lon=17.0)
    faces = [_face(0, img["a.jpg"], 3), _face(1, img["c.jpg"], 3), _face(2, img["d.jpg"], 5)]
    return SimpleNamespace(
        source_directory=None,
        # c is shot before a, so chronological order != id order
        geo_metadata={img["a.jpg"]: t(12), img["b.jpg"]: t(13), img["c.jpg"]: t(9),
                      img["d.jpg"]: t(15)},
        selected_images=[img["a.jpg"], img["c.jpg"]],
        scene_cluster_labels={img["a.jpg"]: 4, img["b.jpg"]: 4, img["c.jpg"]: 1, img["d.jpg"]: 2},
        set_aside=[{"image": img["s.jpg"], "reason": "screenshot (filename)"}],
        composite_scores={img["a.jpg"]: 0.8, img["b.jpg"]: 0.5, img["c.jpg"]: 0.7,
                          img["d.jpg"]: 0.4},
        iqa_scores={img["a.jpg"]: 0.6}, ava_scores={img["a.jpg"]: 5.1}, sharpness_scores={},
        occlusion_scores={}, tilt_angles={}, tilt_confidences={},
        image_captions={img["a.jpg"]: "a bridge"}, straightened_from={},
        step_decisions=[SimpleNamespace(item_id=img["q.jpg"], item_type="image",
                                        step="filter_quality", decision="rejected",
                                        reason="IQA 0.05 < 0.20")],
        face_records=faces,
        people_clusters={3: faces[:2], 5: faces[2:]},
        album_theme={"label": "city trip"},
    )


def fake_pipeline(images, plan):
    return _context(images), SimpleNamespace(success=True, total_duration_ms=1.0, failed_step=None,
                                           error_message=None, step_results=[])


