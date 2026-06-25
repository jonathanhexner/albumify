"""spec-086: backfill image_metric_rows / face_metric_rows from existing runs.

Reads each PipelineResult's blob ``image_metrics`` + the run's Person rows and
writes the normalized tables, reusing PipelineService._write_metric_tables. Lets
existing albums get the new tables without a full pipeline re-run.

Usage: .venv/Scripts/python scripts/backfill_metric_tables.py
"""
import logging

from sim_bench.api.database.session import init_db, get_engine
from sim_bench.api.database.models import PipelineResult, Person, ImageMetricRow, FaceMetricRow
from sim_bench.api.services.pipeline_service import PipelineService
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill")


def main() -> None:
    init_db()
    Session = sessionmaker(bind=get_engine())
    session = Session()
    svc = PipelineService(session)

    results = session.query(PipelineResult).all()
    log.info("Found %d pipeline_results", len(results))

    for res in results:
        run_id = res.run_id
        if not res.image_metrics:
            continue
        # Idempotent: clear any prior rows for this run.
        session.query(ImageMetricRow).filter(ImageMetricRow.run_id == run_id).delete()
        session.query(FaceMetricRow).filter(FaceMetricRow.run_id == run_id).delete()

        people = session.query(Person).filter(Person.run_id == run_id).all()
        svc._write_metric_tables(run_id, res.image_metrics, people)
        n_img = session.query(ImageMetricRow).filter(ImageMetricRow.run_id == run_id).count()
        n_face = session.query(FaceMetricRow).filter(FaceMetricRow.run_id == run_id).count()
        n_link = (
            session.query(FaceMetricRow)
            .filter(FaceMetricRow.run_id == run_id, FaceMetricRow.person_id.isnot(None))
            .count()
        )
        log.info("run %s: %d images, %d faces, %d linked to a person",
                 run_id[:8], n_img, n_face, n_link)

    session.commit()
    log.info("Backfill complete.")


if __name__ == "__main__":
    main()
