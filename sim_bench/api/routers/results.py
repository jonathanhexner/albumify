"""Results router - API endpoints for pipeline results."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from sim_bench.api.database.session import get_session
from sim_bench.api.schemas.result import (
    ResultSummary,
    ResultDetail,
    ImageMetrics,
    ClusterInfo,
    PipelineMetrics,
    ExportRequest,
    ExportResponse,
)
from sim_bench.api.services.result_service import ResultService

router = APIRouter(prefix="/api/v1/results", tags=["results"])


@router.get("/", response_model=list[ResultSummary])
def list_results(
    album_id: Optional[str] = Query(None, description="Filter by album ID"),
    session: Session = Depends(get_session)
):
    """List all completed pipeline results."""
    service = ResultService(session)
    return service.list_results(album_id)


@router.get("/{job_id}", response_model=ResultDetail)
def get_result(job_id: str, session: Session = Depends(get_session)):
    """Get full details of a pipeline result."""
    service = ResultService(session)
    result = service.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.get("/{job_id}/images", response_model=list[ImageMetrics])
def get_result_images(
    job_id: str,
    cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
    person_id: Optional[str] = Query(None, description="Filter by person ID"),
    selected_only: bool = Query(False, description="Only return selected images"),
    session: Session = Depends(get_session)
):
    """Get images from a pipeline result with optional filtering."""
    service = ResultService(session)
    images = service.get_images(
        job_id,
        cluster_id=cluster_id,
        person_id=person_id,
        selected_only=selected_only
    )
    if not images and cluster_id is None and person_id is None and not selected_only:
        # Check if result exists at all
        result = service.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Result not found")
    return images


@router.get("/{job_id}/clusters", response_model=list[ClusterInfo])
def get_result_clusters(job_id: str, session: Session = Depends(get_session)):
    """Get cluster information for a pipeline result."""
    service = ResultService(session)
    clusters = service.get_clusters(job_id)
    if not clusters:
        result = service.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Result not found")
    return clusters


@router.get("/{job_id}/metrics", response_model=PipelineMetrics)
def get_result_metrics(job_id: str, session: Session = Depends(get_session)):
    """Get aggregate metrics for a pipeline result."""
    service = ResultService(session)
    metrics = service.get_metrics(job_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return metrics


@router.get("/{job_id}/comparisons")
def get_siamese_comparisons(job_id: str, session: Session = Depends(get_session)):
    """Get Siamese/duplicate comparison log for debugging and visibility."""
    service = ResultService(session)
    comparisons = service.get_comparisons(job_id)
    if comparisons is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return comparisons


@router.get("/{job_id}/subclusters")
def get_face_subclusters(job_id: str, session: Session = Depends(get_session)):
    """Get face sub-clusters (grouped by scene and face identity)."""
    service = ResultService(session)
    subclusters = service.get_subclusters(job_id)
    if subclusters is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return subclusters


@router.get("/{job_id}/face-embedding")
def get_face_embedding(
    job_id: str,
    image_path: str = Query(..., description="Image file path"),
    face_index: int = Query(0, description="Face index within image"),
    session: Session = Depends(get_session)
):
    """Get face embedding vector for a specific face. Returns 512-dim float array."""
    from sim_bench.api.database.models import UniversalCache
    import numpy as np
    from sim_bench.pipeline.serializers import Serializers

    # Build cache key
    path_normalized = image_path.replace("\\", "/")
    cache_key = f"{path_normalized}:face_{face_index}"

    entry = (
        session.query(UniversalCache)
        .filter(
            UniversalCache.image_path == cache_key,
            UniversalCache.feature_type == "face_embedding",
        )
        .first()
    )

    if entry is None:
        raise HTTPException(status_code=404, detail=f"No cached embedding for {cache_key}")

    embedding = Serializers.numpy_deserialize(entry.data_blob)
    return {
        "face_key": cache_key,
        "embedding": embedding.tolist(),
        "model_name": entry.model_name,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


@router.get("/{job_id}/face-distance")
def get_face_distance(
    job_id: str,
    image_path_a: str = Query(...),
    face_index_a: int = Query(0),
    image_path_b: str = Query(...),
    face_index_b: int = Query(0),
    session: Session = Depends(get_session)
):
    """Compute cosine distance between two face embeddings."""
    from sim_bench.api.database.models import UniversalCache
    import numpy as np
    from sim_bench.pipeline.serializers import Serializers

    def load_embedding(path, face_idx):
        key = f"{path.replace(chr(92), '/')}:face_{face_idx}"
        entry = session.query(UniversalCache).filter(
            UniversalCache.image_path == key,
            UniversalCache.feature_type == "face_embedding",
        ).first()
        if entry is None:
            raise HTTPException(404, f"No embedding for {key}")
        return Serializers.numpy_deserialize(entry.data_blob), key

    emb_a, key_a = load_embedding(image_path_a, face_index_a)
    emb_b, key_b = load_embedding(image_path_b, face_index_b)

    # Normalize and compute cosine distance
    emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
    emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
    cosine_distance = float(1 - np.dot(emb_a, emb_b))

    return {
        "face_a": key_a,
        "face_b": key_b,
        "cosine_distance": round(cosine_distance, 4),
        "verdict": "same_person" if cosine_distance < 0.3 else ("borderline" if cosine_distance < 0.5 else "different"),
    }


@router.post("/{job_id}/export", response_model=ExportResponse)
def export_result(
    job_id: str,
    request: ExportRequest,
    session: Session = Depends(get_session)
):
    """Export pipeline results to a directory.

    Copies or symlinks selected images to the specified output path.
    Can optionally organize by cluster or person.
    """
    service = ResultService(session)

    # Verify result exists
    result = service.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")

    export_result = service.export_results(
        job_id=job_id,
        output_path=request.output_path,
        include_selected=request.include_selected,
        include_all_filtered=request.include_all_filtered,
        organize_by_cluster=request.organize_by_cluster,
        organize_by_person=request.organize_by_person,
        copy_mode=request.copy_mode
    )

    return export_result
