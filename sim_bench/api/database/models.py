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

    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    album = relationship("Album", back_populates="pipeline_runs")
    result = relationship("PipelineResult", back_populates="run", uselist=False, cascade="all, delete-orphan")
    people = relationship("Person", back_populates="run", cascade="all, delete-orphan")


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
