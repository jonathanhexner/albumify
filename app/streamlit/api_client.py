"""HTTP API client for communicating with the FastAPI backend."""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import requests

from .config import get_config, AppConfig
from .models import (
    Album,
    PipelineProgress,
    PipelineResult,
    PipelineStatus,
    Person,
    ClusterInfo,
    ImageInfo,
    StepResult,
    ApiResponse,
    ExportOptions,
)

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """API error with status code and message."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


class ApiClient:
    """Synchronous HTTP client for the FastAPI backend."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.base_url = self.config.api_base_url.rstrip("/")
        self.timeout = self.config.api_timeout_sec
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request and return JSON response."""
        url = self._url(path)
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_data = e.response.json()
                error_msg = error_data.get("detail", error_msg)
            except Exception:
                pass
            raise ApiError(e.response.status_code, error_msg) from e
        except requests.exceptions.RequestException as e:
            raise ApiError(0, f"Connection error: {e}") from e

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """HTTP GET request."""
        return self._request("GET", path, params=params)

    def _post(
        self, path: str, json: Optional[Dict] = None, params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """HTTP POST request."""
        return self._request("POST", path, json=json, params=params)

    def _delete(self, path: str) -> Dict[str, Any]:
        """HTTP DELETE request."""
        return self._request("DELETE", path)

    def _put(self, path: str, json: Optional[Dict] = None) -> Dict[str, Any]:
        """HTTP PUT request."""
        return self._request("PUT", path, json=json)

    # Health check
    def health_check(self) -> bool:
        """Check if the API is available."""
        try:
            response = self._get("/health")
            return response.get("status") == "ok"
        except Exception:
            return False

    # Album operations
    def list_albums(self) -> List[Album]:
        """Get list of all albums."""
        data = self._get("/api/v1/albums/")
        albums = []
        # API returns list directly, not wrapped in {"albums": [...]}
        if isinstance(data, list):
            for item in data:
                albums.append(self._parse_album(item))
        return albums

    def get_album(self, album_id: str) -> Album:
        """Get album by ID."""
        data = self._get(f"/api/v1/albums/{album_id}")
        return self._parse_album(data)

    def create_album(self, name: str, source_directory: str) -> Album:
        """Create new album from source directory."""
        data = self._post(
            "/api/v1/albums/",
            json={"name": name, "source_path": source_directory},
        )
        return self._parse_album(data)

    def delete_album(self, album_id: str) -> bool:
        """Delete album."""
        self._delete(f"/api/v1/albums/{album_id}")
        return True

    # Config profile operations
    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration from pipeline.yaml."""
        return self._get("/api/v1/config/defaults")

    def list_config_profiles(self) -> List[Dict[str, Any]]:
        """List all configuration profiles."""
        data = self._get("/api/v1/config/profiles")
        return data.get("profiles", [])

    def get_config_profile(self, name: str) -> Dict[str, Any]:
        """Get a configuration profile by name."""
        return self._get(f"/api/v1/config/profiles/{name}")

    def create_config_profile(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        base_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new configuration profile."""
        return self._post(
            "/api/v1/config/profiles",
            json={
                "name": name,
                "config": config,
                "description": description,
                "base_profile": base_profile,
            },
        )

    def update_config_profile(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing configuration profile."""
        payload = {}
        if config is not None:
            payload["config"] = config
        if description is not None:
            payload["description"] = description
        return self._put(f"/api/v1/config/profiles/{name}", json=payload)

    def delete_config_profile(self, name: str) -> bool:
        """Delete a configuration profile."""
        self._delete(f"/api/v1/config/profiles/{name}")
        return True

    def get_merged_config(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Get merged configuration for a profile (or default if not specified)."""
        params = {"profile": profile} if profile else None
        return self._get("/api/v1/config/merged", params=params)

    # Pipeline operations
    def start_pipeline(
        self,
        album_id: str,
        steps: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        profile: Optional[str] = None,
        fail_fast: bool = True,
    ) -> str:
        """Start pipeline execution. Returns job ID."""
        data = self._post(
            "/api/v1/pipeline/run",
            json={
                "album_id": album_id,
                "steps": steps,
                "config": config,
                "profile": profile,
                "fail_fast": fail_fast,
            },
        )
        return data.get("job_id", "")

    def get_pipeline_status(self, job_id: str) -> PipelineProgress:
        """Get current pipeline status by job ID."""
        data = self._get(f"/api/v1/pipeline/{job_id}")
        return PipelineProgress(
            status=PipelineStatus(data.get("status", "idle")),
            current_step=data.get("current_step"),
            current_step_progress=data.get("progress", 0.0),
            current_step_message=data.get("message", ""),
            completed_steps=data.get("completed_steps", []),
            total_steps=data.get("total_steps", 0),
        )

    def get_pipeline_result(self, job_id: str) -> Optional[PipelineResult]:
        """Get pipeline result if completed."""
        try:
            data = self._get(f"/api/v1/pipeline/{job_id}/result")
            if not data:
                return None
            return PipelineResult(
                success=True,
                total_duration_ms=data.get("total_duration_ms", 0),
                step_results=[],  # Not available in current API
                error_message=None,
            )
        except ApiError as e:
            if e.status_code == 404:
                return None
            raise

    def stop_pipeline(self, job_id: str) -> bool:
        """Stop running pipeline."""
        # Note: API may not have stop endpoint yet
        return False

    def run_pipeline_with_polling(
        self,
        album_id: str,
        steps: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> PipelineResult:
        """Run pipeline and poll for completion."""
        job_id = self.start_pipeline(album_id, steps, config)

        poll_count = 0
        while poll_count < self.config.max_poll_attempts:
            time.sleep(self.config.poll_interval_sec)
            poll_count += 1

            progress = self.get_pipeline_status(job_id)

            if progress_callback:
                progress_callback(progress)

            if progress.status == PipelineStatus.COMPLETED:
                result = self.get_pipeline_result(job_id)
                if result:
                    return result
                # Result not ready yet, continue polling
                continue

            if progress.status == PipelineStatus.FAILED:
                result = self.get_pipeline_result(job_id)
                if result:
                    return result
                return PipelineResult(
                    success=False,
                    total_duration_ms=0,
                    error_message="Pipeline failed",
                )

        # Timeout
        return PipelineResult(
            success=False,
            total_duration_ms=0,
            error_message="Pipeline timed out",
        )

    # Result/Image operations
    def list_results(self, album_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all completed pipeline results."""
        params = {"album_id": album_id} if album_id else None
        data = self._get("/api/v1/results/", params=params)
        return data if isinstance(data, list) else []

    def get_result_detail(self, job_id: str) -> Dict[str, Any]:
        """Get full details of a pipeline result."""
        return self._get(f"/api/v1/results/{job_id}")

    def get_images(self, job_id: str, cluster_id: Optional[int] = None,
                   person_id: Optional[str] = None,
                   selected_only: bool = False) -> List[ImageInfo]:
        """Get images from a pipeline result with optional filtering."""
        params = {}
        if cluster_id is not None:
            params["cluster_id"] = cluster_id
        if person_id is not None:
            params["person_id"] = person_id
        if selected_only:
            params["selected_only"] = "true"

        data = self._get(f"/api/v1/results/{job_id}/images", params=params)
        return [self._parse_image(img) for img in (data if isinstance(data, list) else [])]

    def get_selected_images(self, job_id: str) -> List[ImageInfo]:
        """Get selected images only."""
        return self.get_images(job_id, selected_only=True)

    # Cluster operations
    def get_clusters(self, job_id: str) -> List[ClusterInfo]:
        """Get all clusters from a pipeline result."""
        data = self._get(f"/api/v1/results/{job_id}/clusters")
        return [self._parse_cluster(c) for c in (data if isinstance(data, list) else [])]

    def get_comparisons(self, job_id: str) -> List[Dict[str, Any]]:
        """Get Siamese/duplicate comparison log for debugging."""
        data = self._get(f"/api/v1/results/{job_id}/comparisons")
        return data if isinstance(data, list) else []

    def get_subclusters(self, job_id: str) -> Dict[str, Any]:
        """Get face sub-clusters (grouped by scene and face identity)."""
        data = self._get(f"/api/v1/results/{job_id}/subclusters")
        return data if isinstance(data, dict) else {}

    # People operations
    def get_people(self, album_id: str, run_id: Optional[str] = None) -> List[Person]:
        """Get all detected people in album."""
        params = {"run_id": run_id} if run_id else None
        data = self._get(f"/api/v1/people/{album_id}", params=params)
        return [self._parse_person(p) for p in (data if isinstance(data, list) else [])]

    def get_person(self, album_id: str, person_id: str) -> Person:
        """Get person by ID."""
        data = self._get(f"/api/v1/people/{album_id}/{person_id}")
        return self._parse_person(data)

    def get_images_by_person(self, album_id: str, person_id: str) -> List[ImageInfo]:
        """Get all images containing a specific person."""
        data = self._get(f"/api/v1/people/{album_id}/{person_id}/images")
        return [self._parse_image(img) for img in (data if isinstance(data, list) else [])]

    def rename_person(self, album_id: str, person_id: str, name: str) -> Person:
        """Rename a person."""
        data = self._request(
            "PATCH",
            f"/api/v1/people/{album_id}/{person_id}",
            json={"name": name},
        )
        return self._parse_person(data)

    def merge_people(self, album_id: str, person_ids: List[str]) -> Person:
        """Merge multiple people into one.

        The first person in the list becomes the merged identity.
        """
        data = self._post(
            f"/api/v1/people/{album_id}/merge",
            json={"person_ids": person_ids},
        )
        return self._parse_person(data)

    def split_person(
        self, album_id: str, person_id: str, face_indices: List[int]
    ) -> List[Person]:
        """Split faces from a person into a new person.

        Returns the updated original person and the new person.
        """
        data = self._post(
            f"/api/v1/people/{album_id}/{person_id}/split",
            json={"face_indices": face_indices},
        )
        if isinstance(data, list):
            return [self._parse_person(p) for p in data]
        return []

    # Export operations
    def export_result(
        self, job_id: str, output_path: str,
        include_selected: bool = True,
        include_all_filtered: bool = False,
        organize_by_cluster: bool = False,
        organize_by_person: bool = False,
        copy_mode: str = "copy"
    ) -> Dict[str, Any]:
        """Export pipeline results to a directory."""
        data = self._post(
            f"/api/v1/results/{job_id}/export",
            json={
                "output_path": output_path,
                "include_selected": include_selected,
                "include_all_filtered": include_all_filtered,
                "organize_by_cluster": organize_by_cluster,
                "organize_by_person": organize_by_person,
                "copy_mode": copy_mode,
            },
        )
        return data

    # Parsing helpers
    def _parse_album(self, data: Dict[str, Any]) -> Album:
        """Parse album from API response."""
        from datetime import datetime

        return Album(
            album_id=data.get("album_id", data.get("id", "")),
            name=data.get("name", ""),
            source_directory=data.get("source_path", data.get("source_directory", "")),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            total_images=data.get("total_images", data.get("image_count", 0)),
            selected_images=data.get("selected_images", 0),
            status=PipelineStatus(data.get("status", "idle")),
        )

    def _parse_image(self, data: Dict[str, Any]) -> ImageInfo:
        """Parse image from API response."""
        return ImageInfo(
            path=data.get("path", ""),
            filename=data.get("filename", Path(data.get("path", "")).name),
            iqa_score=data.get("iqa_score"),
            ava_score=data.get("ava_score"),
            composite_score=data.get("composite_score"),
            face_count=data.get("face_count", 0),
            cluster_id=data.get("cluster_id"),
            is_selected=data.get("is_selected", False),
            sharpness=data.get("sharpness"),
            face_pose_scores=data.get("face_pose_scores"),
            face_eyes_scores=data.get("face_eyes_scores"),
            face_smile_scores=data.get("face_smile_scores"),
        )

    def _parse_cluster(self, data: Dict[str, Any]) -> ClusterInfo:
        """Parse cluster from API response."""
        # API returns images as list of paths (strings), not full image objects
        raw_images = data.get("images", [])
        images = []
        for img in raw_images:
            if isinstance(img, str):
                # Convert path string to basic ImageInfo
                images.append(ImageInfo(
                    path=img,
                    filename=Path(img).name,
                    is_selected=False,
                ))
            elif isinstance(img, dict):
                images.append(self._parse_image(img))

        return ClusterInfo(
            cluster_id=data.get("cluster_id", 0),
            image_count=data.get("image_count", len(images)),
            selected_count=data.get("selected_count", 0),
            has_faces=data.get("has_faces", False),
            face_count=data.get("face_count"),
            images=images,
            person_labels=data.get("person_labels", {}),
        )

    def _parse_person(self, data: Dict[str, Any]) -> Person:
        """Parse person from API response."""
        # API returns thumbnail_image_path, map to representative_face
        representative = data.get("representative_face") or data.get("thumbnail_image_path")
        return Person(
            person_id=data.get("person_id", data.get("id", "")),
            name=data.get("name"),
            face_count=data.get("face_count", 0),
            image_count=data.get("image_count", 0),
            representative_face=representative,
            thumbnail_bbox=data.get("thumbnail_bbox"),
            images=data.get("images", []),
        )


# Global client instance
_client: Optional[ApiClient] = None


def get_client() -> ApiClient:
    """Get the global API client instance."""
    global _client
    if _client is None:
        _client = ApiClient()
    return _client


def set_client(client: ApiClient) -> None:
    """Set the global API client instance."""
    global _client
    _client = client
