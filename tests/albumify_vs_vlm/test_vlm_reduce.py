"""Spec-102 T3 — hierarchical reduce keeps every request under the image cap (413 regression).

The reduce phase used to send the whole shortlist in one request; big trips (Germany, ~320
shortlisted) blew the API's request-size limit (413). These tests mock the API and assert the
shortlist is narrowed in chunks so no single call ever exceeds `max_reduce_images`, and that it
still converges to exactly K.
"""

import re

import sim_bench.albumify_vs_vlm.vlm_arm as va
from sim_bench.albumify_vs_vlm.vlm_arm import VLMArmConfig, _reduce_phase


def _install_fakes(monkeypatch, record: list[int]):
    # no file I/O for images
    monkeypatch.setattr(va, "_img_block", lambda path: {"type": "image", "source": {"data": "x"}})

    def fake_call(client, cfg, content):
        ids = [b["text"][4:] for b in content
               if b.get("type") == "text" and b["text"].startswith("ID: ")]
        record.append(sum(1 for b in content if b.get("type") == "image"))
        prompt = content[-1]["text"]
        if '"order"' in prompt:  # final reduce
            chosen = ids[: cfg.target_k]
            return {"order": chosen,
                    "picks": [{"id": i, "role": "", "reason": ""} for i in chosen]}, (1, 1)
        m = int(re.search(r"up to (\d+)", prompt).group(1))  # map/semifinal shortlist
        return {"shortlist": [{"id": i} for i in ids[:m]]}, (1, 1)

    monkeypatch.setattr(va, "_call", fake_call)


def test_reduce_never_exceeds_image_cap_on_large_shortlist(monkeypatch, tmp_path):
    record: list[int] = []
    _install_fakes(monkeypatch, record)
    shortlist = [f"s{i:03d}" for i in range(320)]  # Germany-sized
    cfg = VLMArmConfig(target_k=20, max_reduce_images=45)
    parsed, _, _ = _reduce_phase(None, cfg, tmp_path, shortlist, set(shortlist))
    assert record, "no calls made"
    assert max(record) <= 45, f"a reduce call sent {max(record)} images (>cap 45) -> would 413"
    assert len(parsed["order"]) == 20


def test_reduce_small_shortlist_single_call(monkeypatch, tmp_path):
    record: list[int] = []
    _install_fakes(monkeypatch, record)
    shortlist = [f"s{i}" for i in range(30)]  # under cap -> one final call, no narrowing
    cfg = VLMArmConfig(target_k=20, max_reduce_images=45)
    parsed, _, _ = _reduce_phase(None, cfg, tmp_path, shortlist, set(shortlist))
    assert len(record) == 1
    assert len(parsed["order"]) == 20


def test_reduce_converges_terminates(monkeypatch, tmp_path):
    record: list[int] = []
    _install_fakes(monkeypatch, record)
    shortlist = [f"s{i:04d}" for i in range(1000)]  # pathological
    cfg = VLMArmConfig(target_k=20, max_reduce_images=45)
    parsed, _, _ = _reduce_phase(None, cfg, tmp_path, shortlist, set(shortlist))
    assert len(parsed["order"]) == 20
    assert max(record) <= 45
