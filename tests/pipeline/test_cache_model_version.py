"""BaseStep cache: a row written under another ``model_version`` is a miss (spec-079, spec-112 TW1.1).

spec-112 Phase A2 keeps the L4 worker's feature cache across runs, so a stale row would now outlive the
container. The version check in ``BaseStep._process_with_cache`` is what stops that; it had no test.
"""

from typing import Any, Dict, List

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import Base
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext


class _CountingStep(BaseStep):
    """Cached step whose output is the version it ran under; counts computed items."""

    def __init__(self, version):
        self._metadata = StepMetadata(name="tw11_counting_step", display_name="Counting",
                                      description="test step", category="analysis")
        self.version = version
        self.computed: List[str] = []
        self.stored: Dict[str, Any] = {}

    def _get_cache_config(self, context, config):
        meta = {} if self.version is None else {"model_version": self.version}
        return {"items": list(context.image_paths), "feature_type": "tw11_feature",
                "model_name": "tw11_model", "metadata": meta}

    def _process_uncached(self, items, context, config):
        self.computed.extend(items)
        return {item: str(self.version) for item in items}

    def _serialize_for_cache(self, result, item):
        return result.encode()

    def _deserialize_from_cache(self, data, item):
        return data.decode()

    def _store_results(self, context, results, config):
        self.stored = dict(results)


@pytest.fixture
def handler():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return UniversalCacheHandler(sessionmaker(bind=engine)())


@pytest.fixture
def images(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(p, "JPEG")
        paths.append(str(p))
    return paths


def _run(step, images, handler):
    step.process(PipelineContext(image_paths=images, cache_handler=handler), {})
    return step


class ut_CacheModelVersion:

    def test_same_version_is_a_hit(self, images, handler):
        _run(_CountingStep("v1"), images, handler)
        again = _run(_CountingStep("v1"), images, handler)
        assert again.computed == []
        assert set(again.stored.values()) == {"v1"}

    def test_other_version_is_a_miss_and_is_replaced(self, images, handler):
        _run(_CountingStep("v1"), images, handler)
        newer = _run(_CountingStep("v2"), images, handler)
        assert sorted(newer.computed) == sorted(images)
        assert set(newer.stored.values()) == {"v2"}
        # the v2 rows replaced the v1 rows: a third v2 run is a pure hit
        assert _run(_CountingStep("v2"), images, handler).computed == []

    def test_unversioned_row_is_a_miss_for_a_versioned_step(self, images, handler):
        _run(_CountingStep(None), images, handler)
        assert sorted(_run(_CountingStep("v1"), images, handler).computed) == sorted(images)

    def test_step_without_a_version_keeps_mtime_only_behaviour(self, images, handler):
        _run(_CountingStep("v1"), images, handler)
        assert _run(_CountingStep(None), images, handler).computed == []
