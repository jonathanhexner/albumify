"""spec-079 — the pipeline-as-config interface: PipelineSpec + validate_spec."""
from __future__ import annotations

from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
from face_cluster.fc_params import FCParams
from sim_bench.pipeline.spec import (
    PipelineSpec,
    PipelineSpecError,
    validate_spec,
    validate_spec_or_raise,
)

FC_V2_PRODUCER = ["discover_images", "detect_persons", "insightface_detect_faces",
                  "detect_face_orientation", "align_faces", "extract_face_embeddings"]


def _fc_v2_spec() -> PipelineSpec:
    return PipelineSpec.from_fcparams(
        FCParams(), producer_steps=FC_V2_PRODUCER, clustering_steps=UNIFIED_CLUSTERING_STEPS,
    )


def ut_FromFcparams_BroadcastsKnobsToClusteringSteps():
    spec = _fc_v2_spec()
    # every clustering step gets the FCParams dump; producer steps get {}.
    for s in UNIFIED_CLUSTERING_STEPS:
        assert spec.step_configs[s]["K"] == FCParams().K
    assert spec.step_configs["detect_persons"] == {}


def ut_ValidSpec_PassesValidation():
    assert validate_spec(_fc_v2_spec()) == []


def ut_MissingMandatoryStep_IsReported():
    # A clustering-only spec never discovers images.
    spec = PipelineSpec(steps=["merge_face_clusters"])
    problems = validate_spec(spec)
    assert any("discover_images" in p for p in problems), problems


def ut_UnknownStep_IsReported():
    spec = PipelineSpec(steps=["discover_images", "no_such_step"])
    problems = validate_spec(spec)
    assert any("no_such_step" in p for p in problems), problems


def ut_InvalidStepParam_IsReported():
    # cluster_people has a typed schema (ClusterPeopleConfig, extra=forbid).
    spec = PipelineSpec(
        steps=["discover_images", "cluster_people"],
        step_configs={"cluster_people": {"definitely_not_a_real_knob": 1}},
    )
    problems = validate_spec(spec)
    assert any("cluster_people" in p for p in problems), problems


def ut_ValidateOrRaise_RaisesOnProblems():
    try:
        validate_spec_or_raise(PipelineSpec(steps=["no_such_step"]))
    except PipelineSpecError as e:
        assert "no_such_step" in str(e)
    else:
        raise AssertionError("expected PipelineSpecError")


def test_pipeline_spec_suite():
    ut_FromFcparams_BroadcastsKnobsToClusteringSteps()
    ut_ValidSpec_PassesValidation()
    ut_MissingMandatoryStep_IsReported()
    ut_UnknownStep_IsReported()
    ut_InvalidStepParam_IsReported()
    ut_ValidateOrRaise_RaisesOnProblems()
