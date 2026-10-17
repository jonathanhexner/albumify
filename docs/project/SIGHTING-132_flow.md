---
title: "SIGHTING-132 - the old people clustering still runs"
---

# SIGHTING-132: the old people clustering still runs

## History

| When | People clustering in `configs/pipeline.yaml` |
|---|---|
| Before spec-079 | One step: `cluster_people` (`sim_bench/pipeline/steps/cluster_people.py`, class `ClusterPeopleStep`) |
| spec-079 | Replaced it with 8 steps ending in `assign_people_clusters` (`sim_bench/pipeline/steps/face_clustering_steps.py`, class `AssignPeopleClustersStep`). `cluster_people` was removed from the yaml. |

## What goes wrong

```
1. pipeline.yaml lists the new steps; cluster_people is not in the list.

2. Three later steps still name the old step as a dependency:
     identity_refinement.py     depends_on = ["cluster_people", ...]
     cluster_by_identity.py     depends_on = ["cluster_people", ...]
     select_best_per_person.py  depends_on = ["cluster_people", ...]

3. sim_bench/pipeline/builder.py  (PipelineBuilder)
   builds the run from the yaml and adds every missing dependency
     -> adds cluster_people back

4. Run:
     assign_people_clusters (new)  ->  context.people_clusters = 80 people
     cluster_people         (old)  ->  context.people_clusters = 118 people   (overwrites)

5. The album shows the 118 people from the old step.
```

Scene clustering (`cluster_scenes`) is not involved.

## Fix (done 2026-10-16)

Deleted `"cluster_people"` from the three `depends_on` lists. Tests in `tests/pipeline/test_builder.py` check that the
default run has no `cluster_people` and runs `cluster_by_identity` before `select_best`. Deleting the step itself:
spec-113.
