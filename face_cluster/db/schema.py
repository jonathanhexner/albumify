"""SQLite schema for the per-run face_clustering.db.

One DDL constant per table so changes to one table produce a focused diff.
`SCHEMA_DDL` is the executescript-ready concatenation; pass it to a fresh
sqlite3 connection at the start of a run.

Bump `SCHEMA_VERSION` for any breaking change (column rename, type change,
constraint that rejects valid prior rows). Additive nullable columns do
NOT need a bump per the spec-033 locked decision.

spec-054: every bump MUST add an entry to ``SCHEMA_HISTORY`` describing
what changed. The arch test ``test_schema_history`` enforces this so the
next person to bump the version is forced to document why.
"""
from __future__ import annotations


# History of schema changes. Keys are the version numbers ever shipped;
# values are one-line human descriptions. SCHEMA_VERSION is derived from
# the max key so adding a new version requires adding a new entry here.
SCHEMA_HISTORY: dict[int, str] = {
    3: (
        "Initial v3 layout — faces, clusters, cluster_assignments, "
        "run_metadata, merge_decisions."
    ),
    4: (
        "Added spec-030 merge fields — run_metadata.parent_run_id, "
        "iteration counters; merge_decisions table refined."
    ),
    5: (
        "spec-040 Phase 4 — added images table; faces gained area_ratio "
        "+ scene_cluster_id; cluster_assignments linked to scenes."
    ),
}

SCHEMA_VERSION: int = max(SCHEMA_HISTORY)


# Allow-list of files produced by RunExporter.export().  Tests assert
# `listdir(run_dir) == EXPECTED_ARTIFACTS` (FR-012).  `crops` is a directory;
# the rest are files.
EXPECTED_ARTIFACTS = (
    "face_clustering.db",
    "embeddings.npy",
    "embedding_face_ids.npy",
    "pipeline_run.json",
    "crops",
)


# ---------------------------------------------------------------------------
# Per-table DDL
# ---------------------------------------------------------------------------

FACES_DDL = """
CREATE TABLE IF NOT EXISTS faces (
    face_id          INTEGER PRIMARY KEY,
    image_path       TEXT,
    image_id         TEXT,
    face_index       INTEGER,
    bbox_x           REAL,
    bbox_y           REAL,
    bbox_w           REAL,
    bbox_h           REAL,
    crop_path        TEXT,
    det_score        REAL,
    blur_score       REAL,
    area             REAL,
    yaw              REAL,
    pitch            REAL,
    roll             REAL,
    is_core          INTEGER NOT NULL,
    rejection_reason TEXT,
    -- spec-033 P-C C-3: image-level context fields plumbed onto the face row.
    -- Nullable for FC App standalone runs (which don't compute image scores)
    -- and for legacy v4 DBs prior to this migration.  No schema version bump
    -- per locked decision (additive only).
    iqa_score        REAL,
    ava_score        REAL,
    sharpness_score  REAL,
    scene_cluster_id INTEGER,
    -- spec-040 Phase 4 (schema v5): canonical unit-normalized geometry.
    -- SIGHTING-064 fix. All *_ratio columns are ∈ [0,1]; the raw bbox_*
    -- and area columns above are deprecated and removed in a follow-up
    -- after one release.
    area_ratio       REAL,
    bbox_x_ratio     REAL,
    bbox_y_ratio     REAL,
    bbox_w_ratio     REAL,
    bbox_h_ratio     REAL
);
"""


FACE_SCORES_DDL = """
CREATE TABLE IF NOT EXISTS face_scores (
    face_id          INTEGER PRIMARY KEY REFERENCES faces(face_id),
    pose_score       REAL,
    eyes_score       REAL,
    expression_score REAL,
    frontal_score    REAL,
    is_clusterable   INTEGER
);
"""


CLUSTERS_DDL = """
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id     INTEGER NOT NULL,
    iteration      INTEGER NOT NULL,
    size           INTEGER NOT NULL,
    diameter       REAL,
    avg_intra_dist REAL,
    origin         TEXT NOT NULL,
    parent_ids     TEXT NOT NULL,
    PRIMARY KEY (cluster_id, iteration)
);
"""


CLUSTER_ASSIGNMENTS_DDL = """
CREATE TABLE IF NOT EXISTS cluster_assignments (
    face_id     INTEGER NOT NULL REFERENCES faces(face_id),
    cluster_id  INTEGER NOT NULL,
    iteration   INTEGER NOT NULL,
    is_exemplar INTEGER NOT NULL DEFAULT 0,
    d10_score   REAL,
    PRIMARY KEY (face_id, iteration)
);
"""


# 28 columns matching MergeDecisionRow.field_names() exactly (FR-004).
# Column order MUST stay aligned with the dataclass so RunExporter can bind
# rows positionally — `test_merge_decisions_column_order_matches_dataclass`
# enforces this.
MERGE_DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS merge_decisions (
    iteration              INTEGER NOT NULL,
    cluster_a              INTEGER NOT NULL,
    cluster_b              INTEGER NOT NULL,
    cluster_a_size         INTEGER NOT NULL,
    cluster_b_size         INTEGER NOT NULL,
    exemplar_dist          REAL    NOT NULL,
    threshold_used         REAL    NOT NULL,
    T_a                    REAL,
    T_b                    REAL,
    T_global               REAL,
    p25_cross_dist         REAL,
    passes_cross           INTEGER,
    support                INTEGER NOT NULL,
    unique_support         INTEGER,
    required_support       INTEGER NOT NULL,
    post_diameter          REAL    NOT NULL,
    max_allowed_diameter   REAL    NOT NULL,
    margin_gap             REAL    NOT NULL,
    margin_dist_to_b       REAL    NOT NULL,
    margin_competitor_dist REAL    NOT NULL,
    margin_competitor_id   INTEGER NOT NULL,
    passes_exemplar        INTEGER NOT NULL,
    passes_support         INTEGER NOT NULL,
    passes_margin          INTEGER NOT NULL,
    passes_diameter        INTEGER NOT NULL,
    action                 TEXT    NOT NULL,
    actually_merged        INTEGER NOT NULL,
    rejection_reason       TEXT,
    PRIMARY KEY (iteration, cluster_a, cluster_b)
);
"""


# spec-032: typed filter-decision log.  One row per (item_id, filter_name).
# Adding a new filter step at runtime needs no schema change; new filter_name
# values just appear as new rows.  Schema is identical regardless of which
# pipeline framework produced the run.
FILTER_DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS filter_decisions (
    item_id       TEXT    NOT NULL,
    item_type     TEXT    NOT NULL,
    parent_id     TEXT,
    filter_name   TEXT    NOT NULL,
    rejected      INTEGER NOT NULL,
    reason        TEXT    NOT NULL,
    measured_json TEXT    NOT NULL,
    PRIMARY KEY (item_id, filter_name)
);
"""


# ---------------------------------------------------------------------------
# spec-040 Phase 4 (schema v5): new tables for image / scene-side persistence.
# Closes SIGHTING-065 (image fields denormalized onto faces) and SIGHTING-066
# (scene side has no structured persistence).
# ---------------------------------------------------------------------------

IMAGES_DDL = """
CREATE TABLE IF NOT EXISTS images (
    image_path        TEXT PRIMARY KEY,
    image_id          TEXT,
    width_px          INTEGER,
    height_px         INTEGER,
    n_faces           INTEGER NOT NULL DEFAULT 0,
    iqa_score         REAL,
    ava_score         REAL,
    sharpness_score   REAL,
    composite_score   REAL,
    scene_cluster_id  INTEGER,
    filter_passed     INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL
);
"""

SCENE_CLUSTERS_DDL = """
CREATE TABLE IF NOT EXISTS scene_clusters (
    scene_cluster_id     INTEGER NOT NULL,
    iteration            INTEGER NOT NULL,
    size                 INTEGER NOT NULL,
    method               TEXT NOT NULL,
    exemplar_image_path  TEXT,
    avg_intra_distance   REAL,
    created_at           TEXT NOT NULL,
    PRIMARY KEY (scene_cluster_id, iteration)
);
"""

SCENE_CLUSTER_ASSIGNMENTS_DDL = """
CREATE TABLE IF NOT EXISTS scene_cluster_assignments (
    image_path           TEXT NOT NULL REFERENCES images(image_path),
    scene_cluster_id     INTEGER NOT NULL,
    iteration            INTEGER NOT NULL,
    distance_to_centroid REAL,
    PRIMARY KEY (image_path, iteration)
);
"""


RUN_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS run_metadata (
    run_id                  TEXT PRIMARY KEY,
    source_album            TEXT NOT NULL,
    producer                TEXT NOT NULL,
    parent_run_id           TEXT,
    config_json             TEXT NOT NULL,
    merge_thresholds_json   TEXT,
    merge_iter_summary_json TEXT,
    n_images                INTEGER NOT NULL,
    n_faces                 INTEGER NOT NULL,
    n_core                  INTEGER NOT NULL,
    n_clusters_base         INTEGER NOT NULL,
    n_clusters_final        INTEGER NOT NULL,
    n_merges                INTEGER NOT NULL,
    n_iterations            INTEGER NOT NULL,
    started_at              TEXT NOT NULL,
    finished_at             TEXT NOT NULL,
    schema_version          INTEGER NOT NULL
);
"""


INDEXES_DDL = """
CREATE INDEX IF NOT EXISTS idx_fd_filter      ON filter_decisions(filter_name);
CREATE INDEX IF NOT EXISTS idx_fd_item        ON filter_decisions(item_type, item_id);
CREATE INDEX IF NOT EXISTS idx_assign_iter    ON cluster_assignments(iteration);
CREATE INDEX IF NOT EXISTS idx_assign_cluster ON cluster_assignments(cluster_id, iteration);
CREATE INDEX IF NOT EXISTS idx_md_iter        ON merge_decisions(iteration);
CREATE INDEX IF NOT EXISTS idx_md_pair        ON merge_decisions(cluster_a, cluster_b);
CREATE INDEX IF NOT EXISTS idx_sca_cluster    ON scene_cluster_assignments(scene_cluster_id, iteration);
CREATE INDEX IF NOT EXISTS idx_images_scene   ON images(scene_cluster_id);
"""


# Concatenated DDL for executescript().  Order matters only where foreign
# keys reference an earlier table — `faces` must precede `face_scores` and
# `cluster_assignments`.
SCHEMA_DDL = "\n".join([
    "PRAGMA foreign_keys = ON;",
    FACES_DDL,
    FACE_SCORES_DDL,
    IMAGES_DDL,                       # spec-040 v5: must precede scene_cluster_assignments (FK ref)
    SCENE_CLUSTERS_DDL,
    SCENE_CLUSTER_ASSIGNMENTS_DDL,
    CLUSTERS_DDL,
    CLUSTER_ASSIGNMENTS_DDL,
    MERGE_DECISIONS_DDL,
    FILTER_DECISIONS_DDL,
    RUN_METADATA_DDL,
    INDEXES_DDL,
])
