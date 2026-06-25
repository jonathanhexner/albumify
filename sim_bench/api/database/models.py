"""SQLAlchemy database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Index, LargeBinary, UniqueConstraint
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class Album(Base):
    """An uploaded photo album."""
    __tablename__ = "albums"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    image_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    pipeline_runs = relationship("PipelineRun", back_populates="album", cascade="all, delete-orphan")
    people = relationship("Person", back_populates="album", cascade="all, delete-orphan")
    events = relationship("UserEvent", back_populates="album", cascade="all, delete-orphan")
    face_overrides = relationship("FaceOverride", back_populates="album", cascade="all, delete-orphan")


class PipelineRun(Base):
    """A single pipeline execution."""
    __tablename__ = "pipeline_runs"

    id = Column(String, primary_key=True)
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)

    pipeline_name = Column(String)
    steps = Column(JSON)
    step_configs = Column(JSON)
    fail_fast = Column(Boolean, default=True)

    status = Column(String, default="pending")
    current_step = Column(String, nullable=True)
    progress = Column(Float, default=0.0)
    error_message = Column(String, nullable=True)
    completed_steps = Column(JSON, nullable=True)  # [{step, duration_ms, status}]

    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    album = relationship("Album", back_populates="pipeline_runs")
    result = relationship("PipelineResult", back_populates="run", uselist=False, cascade="all, delete-orphan")
    people = relationship("Person", back_populates="run", cascade="all, delete-orphan")
    events = relationship("UserEvent", back_populates="run", cascade="all, delete-orphan")
    face_overrides = relationship("FaceOverride", back_populates="run", cascade="all, delete-orphan")


class PipelineResult(Base):
    """Results of a completed pipeline run."""
    __tablename__ = "pipeline_results"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=False)

    total_images = Column(Integer)
    filtered_images = Column(Integer)
    num_clusters = Column(Integer)
    num_selected = Column(Integer)

    scene_clusters = Column(JSON)
    face_subclusters = Column(JSON)  # Sub-clusters by face identity within scenes
    selected_images = Column(JSON)
    image_metrics = Column(JSON)
    siamese_comparisons = Column(JSON)  # Siamese/duplicate comparison log for debugging

    step_timings = Column(JSON)
    total_duration_ms = Column(Integer)
    fc_export_dir = Column(String, nullable=True)  # Face clustering export path for standalone app
    step_decisions = Column(JSON, nullable=True)  # Per-item decision records from pipeline steps

    run = relationship("PipelineRun", back_populates="result")


class Person(Base):
    """A detected person (cluster of faces) in an album."""
    __tablename__ = "people"

    id = Column(String, primary_key=True)
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=False)

    person_index = Column(Integer)  # 0, 1, 2, ... (cluster ID)
    name = Column(String, nullable=True)  # User-assigned name

    # Thumbnail (best face for this person)
    thumbnail_image_path = Column(String)
    thumbnail_face_index = Column(Integer)
    thumbnail_bbox = Column(JSON)

    # Statistics
    face_count = Column(Integer, default=0)
    image_count = Column(Integer, default=0)

    # All face instances for this person
    # [{image_path, face_index, bbox, score}, ...]
    face_instances = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)

    album = relationship("Album", back_populates="people")
    run = relationship("PipelineRun", back_populates="people")
    face_overrides = relationship("FaceOverride", back_populates="person")

    __table_args__ = (
        Index('idx_person_album', 'album_id'),
        Index('idx_person_run', 'run_id'),
    )


class ConfigProfile(Base):
    """Named configuration profile for pipeline settings."""
    __tablename__ = "config_profiles"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(String, nullable=True)
    config = Column(JSON, nullable=False)  # Full pipeline config dict
    is_default = Column(Boolean, default=False)  # Mark one as the default
    is_system = Column(Boolean, default=False)  # True = managed by system (from YAML)
    user_id = Column(String, nullable=True, index=True)  # For user-specific profiles
    parent_profile_id = Column(String, ForeignKey("config_profiles.id"), nullable=True)  # Inheritance
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UniversalCache(Base):
    """Universal cache with metadata + flexible data blob storage."""
    __tablename__ = "universal_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Cache key (unique together)
    image_path = Column(String, nullable=False, index=True)
    feature_type = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=True)

    # Flexible data storage (opaque bytes)
    data_blob = Column(LargeBinary, nullable=False)

    # Metadata for invalidation and housekeeping
    image_mtime = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_universal_lookup', 'image_path', 'feature_type', 'model_name'),
        UniqueConstraint('image_path', 'feature_type', 'model_name', name='uq_universal_cache_key'),
    )


class UserEvent(Base):
    """Generic event tracking for user actions, feedback, AI requests."""
    __tablename__ = "user_events"

    id = Column(String, primary_key=True)
    album_id = Column(String, ForeignKey("albums.id"), nullable=True)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=True)

    event_type = Column(String, nullable=False)
    event_data = Column(JSON, nullable=False)

    status = Column(String, default="completed")
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)

    source = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)

    is_undone = Column(Boolean, default=False)
    undone_by_id = Column(String, ForeignKey("user_events.id"), nullable=True)

    album = relationship("Album", back_populates="events")
    run = relationship("PipelineRun", back_populates="events")

    __table_args__ = (
        Index('idx_event_album', 'album_id'),
        Index('idx_event_run', 'run_id'),
        Index('idx_event_type', 'event_type'),
    )


class FaceOverride(Base):
    """Persistent face classification overrides from user corrections."""
    __tablename__ = "face_overrides"

    id = Column(String, primary_key=True)
    album_id = Column(String, ForeignKey("albums.id"), nullable=False)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=True)

    # Face identifier: "image_path:face_N"
    face_key = Column(String, nullable=False)

    # Classification status
    status = Column(String, nullable=False)  # "assigned", "untagged", "not_a_face"
    person_id = Column(String, ForeignKey("people.id"), nullable=True)

    # For "not_a_face" - store embedding for similarity rejection
    embedding = Column(JSON, nullable=True)

    # Audit
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, default="user")

    # Relationships
    album = relationship("Album", back_populates="face_overrides")
    run = relationship("PipelineRun", back_populates="face_overrides")
    person = relationship("Person", back_populates="face_overrides")

    __table_args__ = (
        Index('idx_face_override_album_face', 'album_id', 'face_key'),
        Index('idx_face_override_status', 'status'),
        UniqueConstraint('album_id', 'face_key', name='uq_face_override_album_face'),
    )


# ---------------------------------------------------------------------------
# spec-086 (slice 1): normalized per-image / per-face metric tables.
#
# These replace the JSON-blob stitch (pipeline_results.image_metrics +
# people.face_instances) for the People & Faces image listing. Written by
# PipelineService alongside the blob (dual-write), read by ImageRepository via a
# real SQL JOIN. Field set mirrors the ImageMetrics API contract (spec-085) so
# the repository can rebuild ImageMetrics without re-deriving anything.
# ---------------------------------------------------------------------------

class ImageMetricRow(Base):
    """One row per image in a run. Scalar per-image metrics (no per-face lists)."""

    __tablename__ = "image_metric_rows"

    run_id = Column(String, ForeignKey("pipeline_runs.id"), primary_key=True)
    image_path = Column(String, primary_key=True)

    iqa_score = Column(Float, nullable=True)
    ava_score = Column(Float, nullable=True)
    sharpness = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    person_penalty = Column(Float, nullable=True)
    cluster_id = Column(Integer, nullable=True)
    face_count = Column(Integer, default=0)
    is_selected = Column(Boolean, default=False)
    filter_reason = Column(String, nullable=True)

    # InsightFace person detection
    person_detected = Column(Boolean, nullable=True)
    body_facing_score = Column(Float, nullable=True)
    person_confidence = Column(Float, nullable=True)
    best_frontal_score = Column(Float, nullable=True)
    best_centrality = Column(Float, nullable=True)

    __table_args__ = (
        Index('idx_imgmetric_run', 'run_id'),
    )


class FaceMetricRow(Base):
    """One row per detected face. Carries bbox + per-face scores + person link."""

    __tablename__ = "face_metric_rows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=False)
    image_path = Column(String, nullable=False)
    face_index = Column(Integer, nullable=False)
    # person_id is the Person.id this face was clustered into (None if unassigned).
    person_id = Column(String, ForeignKey("people.id"), nullable=True)

    # Bounding box — both normalized [0,1] and pixel, as the UI overlay accepts either.
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_w = Column(Float, nullable=True)
    bbox_h = Column(Float, nullable=True)
    bbox_x_px = Column(Float, nullable=True)
    bbox_y_px = Column(Float, nullable=True)
    bbox_w_px = Column(Float, nullable=True)
    bbox_h_px = Column(Float, nullable=True)

    confidence = Column(Float, nullable=True)
    filter_passed = Column(Boolean, default=True)
    bbox_ratio = Column(Float, nullable=True)
    relative_size = Column(Float, nullable=True)
    eye_ratio = Column(Float, nullable=True)

    pose_score = Column(Float, nullable=True)
    eyes_score = Column(Float, nullable=True)
    smile_score = Column(Float, nullable=True)
    roll_angle = Column(Float, nullable=True)

    __table_args__ = (
        Index('idx_facemetric_run_person', 'run_id', 'person_id'),
        Index('idx_facemetric_run_image', 'run_id', 'image_path'),
    )
