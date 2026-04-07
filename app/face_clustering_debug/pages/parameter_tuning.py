"""Parameter tuning page — re-run clustering with adjusted parameters."""

import pandas as pd
import streamlit as st

from app.face_clustering_debug.components.face_grid import render_face_grid
from app.face_clustering_debug.components.param_sliders import render_param_sliders
from app.face_clustering_debug.models.schemas import ClusteringRequest
from app.face_clustering_debug.services.clustering_runner import ClusteringRunner
from app.face_clustering_debug.services.protocols import DataLoaderProtocol


def render_parameter_tuning_page(loader: DataLoaderProtocol) -> None:
    st.header("⚙️ Parameter Tuning")
    st.caption("Adjust algorithm parameters and re-run clustering live. "
               "Results are displayed immediately without saving to disk.")

    algorithm = st.selectbox("🧮 Algorithm", ClusteringRunner.get_available_algorithms())
    param_defs = ClusteringRunner.get_algorithm_params(algorithm)

    with st.expander("🎛️ Parameters", expanded=True):
        params = render_param_sliders(param_defs)

    if not st.button("▶️ Run Clustering", type="primary"):
        return

    embeddings = loader.load_embeddings()
    faces = loader.load_faces()

    with st.spinner("Running clustering…"):
        request = ClusteringRequest(algorithm=algorithm, params=params,
                                    embeddings=embeddings, faces=faces)
        result = ClusteringRunner.run(request)

    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Clusters", result.n_clusters)
    m2.metric("🔴 Noise", result.n_noise)
    m3.metric("👤 Faces", len(result.faces))

    rows = [{"Cluster": c.cluster_id, "Faces": len(c.face_indices),
             "Threshold T": round(c.threshold, 3)} for c in result.clusters]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    faces_by_idx = {f.index: f for f in result.faces}
    for cluster in result.clusters:
        with st.expander(f"Cluster {cluster.cluster_id}  ({len(cluster.face_indices)} faces, T={cluster.threshold:.3f})"):
            cluster_faces = [faces_by_idx[i] for i in cluster.face_indices if i in faces_by_idx]
            render_face_grid(faces=cluster_faces, get_crop_fn=loader.get_face_crop,
                             highlight_indices=cluster.exemplar_indices, columns=10,
                             key_prefix=f"pt_c{cluster.cluster_id}_")
