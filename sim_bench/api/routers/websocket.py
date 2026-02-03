"""WebSocket router - real-time progress updates."""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sim_bench.api.database.session import get_session_direct
from sim_bench.api.services.pipeline_service import PipelineService

router = APIRouter()


@router.websocket("/ws/progress/{job_id}")
async def pipeline_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time pipeline progress.

    Sends JSON messages:
    - {"type": "progress", "step": "score_iqa", "progress": 0.45, "message": "..."}
    - {"type": "complete"}
    - {"type": "error", "message": "..."}
    """
    await websocket.accept()

    session = get_session_direct()
    service = PipelineService(session)
    queue = None

    try:
        queue = service.subscribe(job_id)

        while True:
            try:
                update = await asyncio.wait_for(queue.get(), timeout=30.0)

                await websocket.send_json({
                    "type": "progress",
                    "step": update.step,
                    "progress": update.progress,
                    "message": update.message
                })

            except asyncio.TimeoutError:
                run = service.get_status(job_id)
                if run and run.status in ("completed", "failed"):
                    if run.status == "completed":
                        await websocket.send_json({"type": "complete"})
                    else:
                        await websocket.send_json({"type": "error", "message": run.error_message})
                    break

                await websocket.send_json({"type": "ping"})

    except ValueError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        if queue is not None:
            service.unsubscribe(job_id, queue)
        session.close()
