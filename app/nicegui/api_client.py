"""API client for communicating with the FastAPI backend."""

import asyncio
from typing import Callable, Optional
import httpx
import websockets


class APIClient:
    """HTTP and WebSocket client for the album organization API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url
        self._ws_url = base_url.replace("http", "ws")

    async def create_album(self, name: str, source_path: str) -> dict:
        """Create a new album."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/v1/albums/",
                json={"name": name, "source_path": source_path}
            )
            response.raise_for_status()
            return response.json()

    async def list_albums(self) -> list[dict]:
        """List all albums."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/api/v1/albums/")
            response.raise_for_status()
            return response.json()

    async def get_album(self, album_id: str) -> dict:
        """Get an album by ID."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/api/v1/albums/{album_id}")
            response.raise_for_status()
            return response.json()

    async def delete_album(self, album_id: str) -> bool:
        """Delete an album."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{self._base_url}/api/v1/albums/{album_id}")
            response.raise_for_status()
            return True

    async def list_steps(self, category: Optional[str] = None) -> list[dict]:
        """List all available pipeline steps."""
        async with httpx.AsyncClient() as client:
            params = {"category": category} if category else {}
            response = await client.get(f"{self._base_url}/api/v1/steps/", params=params)
            response.raise_for_status()
            return response.json()["steps"]

    async def run_pipeline(
        self,
        album_id: str,
        steps: Optional[list[str]] = None,
        config: Optional[dict] = None
    ) -> str:
        """Start a pipeline run. Returns job_id."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/v1/pipeline/run",
                json={"album_id": album_id, "steps": steps, "config": config}
            )
            response.raise_for_status()
            return response.json()["job_id"]

    async def get_pipeline_status(self, job_id: str) -> dict:
        """Get pipeline status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/api/v1/pipeline/{job_id}")
            response.raise_for_status()
            return response.json()

    async def get_pipeline_result(self, job_id: str) -> dict:
        """Get pipeline result."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/api/v1/pipeline/{job_id}/result")
            response.raise_for_status()
            return response.json()

    async def subscribe_progress(
        self,
        job_id: str,
        on_progress: Callable[[dict], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None]
    ) -> None:
        """Subscribe to pipeline progress via WebSocket."""
        uri = f"{self._ws_url}/ws/progress/{job_id}"

        async with websockets.connect(uri) as ws:
            while True:
                message = await ws.recv()
                import json
                data = json.loads(message)

                if data["type"] == "progress":
                    on_progress(data)
                elif data["type"] == "complete":
                    on_complete()
                    break
                elif data["type"] == "error":
                    on_error(data.get("message", "Unknown error"))
                    break
                elif data["type"] == "ping":
                    continue

    # ============ Results API ============

    async def list_results(self, album_id: Optional[str] = None) -> list[dict]:
        """List all completed pipeline results."""
        async with httpx.AsyncClient() as client:
            params = {"album_id": album_id} if album_id else {}
            response = await client.get(f"{self._base_url}/api/v1/results/", params=params)
            response.raise_for_status()
            return response.json()

    async def get_result(self, job_id: str) -> dict:
        """Get full result details."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/api/v1/results/{job_id}")
            response.raise_for_status()
            return response.json()

    async def get_result_images(
        self,
        job_id: str,
        cluster_id: Optional[int] = None,
        person_id: Optional[str] = None,
        selected_only: bool = False
    ) -> list[dict]:
        """Get images from a result with optional filtering."""
        async with httpx.AsyncClient() as client:
            params = {"selected_only": selected_only}
            if cluster_id is not None:
                params["cluster_id"] = cluster_id
            if person_id:
                params["person_id"] = person_id
            response = await client.get(
                f"{self._base_url}/api/v1/results/{job_id}/images",
                params=params
            )
            response.raise_for_status()
            return response.json()

    async def get_result_clusters(self, job_id: str) -> list[dict]:
        """Get cluster information."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/api/v1/results/{job_id}/clusters")
            response.raise_for_status()
            return response.json()

    async def get_result_metrics(self, job_id: str) -> dict:
        """Get pipeline metrics."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self._base_url}/api/v1/results/{job_id}/metrics")
            response.raise_for_status()
            return response.json()

    async def export_result(
        self,
        job_id: str,
        output_path: str,
        include_selected: bool = True,
        include_all_filtered: bool = False,
        organize_by_cluster: bool = False,
        organize_by_person: bool = False,
        copy_mode: str = "copy"
    ) -> dict:
        """Export results to a directory."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/v1/results/{job_id}/export",
                json={
                    "output_path": output_path,
                    "include_selected": include_selected,
                    "include_all_filtered": include_all_filtered,
                    "organize_by_cluster": organize_by_cluster,
                    "organize_by_person": organize_by_person,
                    "copy_mode": copy_mode
                }
            )
            response.raise_for_status()
            return response.json()

    # ============ People API ============

    async def list_people(self, album_id: str, run_id: Optional[str] = None) -> list[dict]:
        """List all people detected in an album."""
        async with httpx.AsyncClient() as client:
            params = {"run_id": run_id} if run_id else {}
            response = await client.get(
                f"{self._base_url}/api/v1/people/{album_id}",
                params=params
            )
            response.raise_for_status()
            return response.json()

    async def get_person(self, album_id: str, person_id: str) -> dict:
        """Get details of a specific person."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._base_url}/api/v1/people/{album_id}/{person_id}"
            )
            response.raise_for_status()
            return response.json()

    async def get_person_images(self, album_id: str, person_id: str) -> list[dict]:
        """Get all images containing a person."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._base_url}/api/v1/people/{album_id}/{person_id}/images"
            )
            response.raise_for_status()
            return response.json()

    async def rename_person(self, album_id: str, person_id: str, name: str) -> dict:
        """Rename a person."""
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self._base_url}/api/v1/people/{album_id}/{person_id}",
                json={"name": name}
            )
            response.raise_for_status()
            return response.json()

    async def merge_people(self, album_id: str, person_ids: list[str]) -> dict:
        """Merge multiple people into one."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/v1/people/{album_id}/merge",
                json={"person_ids": person_ids}
            )
            response.raise_for_status()
            return response.json()

    async def split_person(
        self,
        album_id: str,
        person_id: str,
        face_indices: list[int]
    ) -> list[dict]:
        """Split faces from a person into a new person."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/v1/people/{album_id}/{person_id}/split",
                json={"face_indices": face_indices}
            )
            response.raise_for_status()
            return response.json()


api_client = APIClient()
