"""
Streamlit app for manually labeling face clusters.

Allows users to assign corrected_identity to each cluster for ML training.

Usage:
    streamlit run app/face_clustering_labeling.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import json
from datetime import datetime

st.set_page_config(
    page_title="Face Cluster Labeling",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Face Cluster Labeling")
st.markdown("Assign corrected identities to clusters for ML training")

# Session state initialization
if 'export_dir' not in st.session_state:
    st.session_state.export_dir = None
if 'faces_df' not in st.session_state:
    st.session_state.faces_df = None
if 'clusters_df' not in st.session_state:
    st.session_state.clusters_df = None
if 'labels_df' not in st.session_state:
    st.session_state.labels_df = None
if 'identity_suggestions' not in st.session_state:
    st.session_state.identity_suggestions = []


@st.cache_data
def load_csvs(export_dir: Path):
    """Load exported CSV files."""
    faces_df = pd.read_csv(export_dir / 'faces.csv')
    clusters_df = pd.read_csv(export_dir / 'clusters.csv')

    # Try to load existing labels
    labels_path = export_dir / 'corrected_labels.csv'
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
    else:
        # Initialize with cluster IDs as default identities
        labels_df = pd.DataFrame({
            'cluster_id': clusters_df['cluster_id'],
            'corrected_identity': [f"person_{i}" for i in clusters_df['cluster_id']]
        })

    return faces_df, clusters_df, labels_df


def get_cluster_face_images(cluster_id: int, faces_df: pd.DataFrame, crops_dir: Path, max_faces: int = None):
    """Load face crop images for a cluster.

    Args:
        cluster_id: Cluster ID to load faces for
        faces_df: DataFrame with face metadata
        crops_dir: Directory containing face crop images
        max_faces: Maximum number of faces to load (None = load all)
    """
    cluster_faces = faces_df[faces_df['cluster_id'] == cluster_id]
    if max_faces is not None:
        cluster_faces = cluster_faces.head(max_faces)

    images = []

    for _, face_row in cluster_faces.iterrows():
        face_id = int(face_row['face_id'])  # Convert to int

        # Try to load face crop
        for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
            crop_path = crops_dir / pattern
            if crop_path.exists():
                try:
                    img = Image.open(crop_path)
                    images.append((face_id, img))
                    break
                except Exception as e:
                    st.warning(f"Failed to load {crop_path}: {e}")

    return images


@st.dialog("View All Faces in Cluster")
def show_all_cluster_faces(cluster_id: int, cluster_size: int, faces_df: pd.DataFrame, crops_dir: Path):
    """Modal dialog showing all faces in a cluster."""
    st.subheader(f"Cluster {cluster_id} - All {cluster_size} Faces")

    # Load all face images
    images = get_cluster_face_images(cluster_id, faces_df, crops_dir, max_faces=None)

    if len(images) == 0:
        st.warning(f"No face crops found for cluster {cluster_id}")
        return

    st.info(f"Showing {len(images)} of {cluster_size} faces (some may be missing from disk)")

    # Display in grid with more columns for modal view
    cols_per_row = 8
    n_rows = (len(images) + cols_per_row - 1) // cols_per_row

    for row in range(n_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx
            if img_idx < len(images):
                face_id, img = images[img_idx]
                with cols[col_idx]:
                    st.image(img, caption=f"{face_id}", use_container_width=True)


@st.dialog("View All Faces in Pre-merge Cluster")
def show_all_pre_merge_cluster_faces(pre_cluster_id: int, cluster_size: int, faces_df: pd.DataFrame, crops_dir: Path):
    """Modal dialog showing all faces in a pre-merge cluster."""
    st.subheader(f"Pre-merge Cluster {pre_cluster_id} - All {cluster_size} Faces")

    # Load all face images from pre-merge cluster
    pre_cluster_faces = faces_df[faces_df['pre_merge_cluster_id'] == pre_cluster_id]

    images = []
    for _, face_row in pre_cluster_faces.iterrows():
        face_id = int(face_row['face_id'])  # Convert to int
        for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
            crop_path = crops_dir / pattern
            if crop_path.exists():
                try:
                    img = Image.open(crop_path)
                    images.append((face_id, img))
                    break
                except Exception as e:
                    st.warning(f"Failed to load {crop_path}: {e}")

    if len(images) == 0:
        st.warning(f"No face crops found for pre-merge cluster {pre_cluster_id}")
        return

    st.info(f"Showing {len(images)} of {cluster_size} faces (some may be missing from disk)")

    # Display in grid with more columns for modal view
    cols_per_row = 8
    n_rows = (len(images) + cols_per_row - 1) // cols_per_row

    for row in range(n_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx
            if img_idx < len(images):
                face_id, img = images[img_idx]
                with cols[col_idx]:
                    st.image(img, caption=f"{face_id}", use_container_width=True)


def show_merge_analysis(
    merge_decisions_df: pd.DataFrame,
    faces_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    crops_dir: Path,
    cluster_lineage: dict,
    pre_merge_clusters_df: pd.DataFrame
):
    """Tab 2: Cluster-centric merge analysis."""
    st.header("🔍 Cluster Merge Analysis")

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        n_merged = len(merge_decisions_df[merge_decisions_df['action'] == 'merged'])
        st.metric("Total Merges", n_merged)
    with col2:
        n_post_clusters = len(clusters_df)
        st.metric("Final Clusters", n_post_clusters)
    with col3:
        n_pre_clusters = len(pre_merge_clusters_df)
        st.metric("Initial Clusters", n_pre_clusters)

    st.markdown("---")

    # Step 1: Select cluster to debug
    st.subheader("Step 1: Select Cluster to Debug")

    # Get clusters sorted by size
    sorted_clusters = clusters_df.sort_values('cluster_size', ascending=False)

    selected_cluster = st.selectbox(
        "Choose post-merge cluster",
        options=sorted_clusters['cluster_id'].tolist(),
        format_func=lambda cid: f"Cluster {cid} ({clusters_df[clusters_df['cluster_id']==cid].iloc[0]['cluster_size']} faces)"
    )

    if selected_cluster is None:
        st.info("Select a cluster to see its merge history")
        return

    st.markdown("---")

    # Step 2: Show cluster formation
    st.subheader(f"Step 2: Formation History of Cluster {selected_cluster}")

    # Get lineage
    source_clusters = cluster_lineage.get(str(selected_cluster), [])

    if len(source_clusters) <= 1:
        st.info(f"Cluster {selected_cluster} was not formed by merging (no merge history)")
        return

    # Show summary
    st.markdown(f"**Cluster {selected_cluster} was formed from {len(source_clusters)} pre-merge clusters:**")
    st.caption(", ".join([f"Pre-cluster {c}" for c in sorted(source_clusters)]))

    # Show original pre-cluster (same ID as post-cluster)
    if selected_cluster in source_clusters:
        st.markdown("---")
        st.markdown(f"### Original Pre-merge Cluster {selected_cluster}")
        st.caption("This is the 'base' cluster that other clusters merged into")

        original_faces = faces_df[faces_df['pre_merge_cluster_id'] == selected_cluster]
        total_original = len(original_faces)

        if total_original > 0:
            images = []
            for _, face_row in original_faces.head(10).iterrows():
                face_id = int(face_row['face_id'])  # Convert to int
                for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
                    crop_path = crops_dir / pattern
                    if crop_path.exists():
                        images.append((face_id, Image.open(crop_path)))
                        break

            if images:
                cols = st.columns(min(len(images), 10))
                for idx, (face_id, img) in enumerate(images):
                    with cols[idx]:
                        st.image(img, caption=f"{face_id}", use_container_width=True)

                if total_original > 10:
                    st.caption(f"Showing 10 of {total_original} faces")
                    if st.button(f"🔍 View All {total_original} Faces", key=f"view_all_original_{selected_cluster}"):
                        show_all_pre_merge_cluster_faces(selected_cluster, total_original, faces_df, crops_dir)

    st.markdown("---")

    # Build merge history for this cluster
    # Track which pre-cluster merged into which post-cluster (as clusters get renumbered)
    merge_history = []

    # Get all merges where cluster_a or result involves our target cluster
    # IMPORTANT: Filter by actually_merged=True to get only executed merges (not just valid candidates)
    actually_merged = merge_decisions_df[
        (merge_decisions_df['action'] == 'merged') &
        (merge_decisions_df['actually_merged'] == True)
    ]

    for _, decision in actually_merged.iterrows():
        cluster_a = decision['cluster_a']
        cluster_b = decision['cluster_b']

        # Check if this merge affected our target cluster
        # The cluster that "survives" is cluster_a
        # So if cluster_a == selected_cluster, this merge added cluster_b into it
        if cluster_a == selected_cluster:
            merge_history.append({
                'iteration': decision['iteration'],
                'absorbed_cluster': cluster_b,
                'into_cluster': cluster_a,
                'exemplar_dist': decision['exemplar_dist'],
                'threshold': decision['threshold_used'],
                'support': decision['support'],
                'decision': decision
            })

    # Sort by iteration
    merge_history.sort(key=lambda x: x['iteration'])

    st.markdown(f"**{len(merge_history)} merges built this cluster:**")

    # Show each merge
    for i, merge in enumerate(merge_history, 1):
        with st.expander(
            f"Merge #{i} (Iteration {merge['iteration']}): "
            f"Pre-cluster {merge['absorbed_cluster']} → Cluster {merge['into_cluster']}",
            expanded=i <= 3  # Expand first 3 by default
        ):
            # Show merge details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Exemplar Distance", f"{merge['exemplar_dist']:.3f}")
            with col2:
                st.metric("Threshold", f"{merge['threshold']:.3f}")
            with col3:
                st.metric("Support", merge['support'])

            # Show evidence checks
            decision = merge['decision']
            checks = []
            if decision['passes_exemplar']:
                checks.append("✅ Exemplar")
            else:
                checks.append("❌ Exemplar")

            if decision['passes_support']:
                checks.append("✅ Support")
            else:
                checks.append("❌ Support")

            if decision['passes_margin']:
                checks.append("✅ Margin")
            else:
                checks.append("❌ Margin")

            if decision['passes_diameter']:
                checks.append("✅ Diameter")
            else:
                checks.append("❌ Diameter")

            st.caption(" | ".join(checks))

            # Show faces from absorbed cluster
            st.markdown(f"#### Faces from Pre-cluster {merge['absorbed_cluster']}")

            absorbed_faces = faces_df[faces_df['pre_merge_cluster_id'] == merge['absorbed_cluster']].head(10)

            if len(absorbed_faces) > 0:
                images = []
                for _, face_row in absorbed_faces.iterrows():
                    face_id = int(face_row['face_id'])  # Convert to int
                    for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
                        crop_path = crops_dir / pattern
                        if crop_path.exists():
                            images.append((face_id, Image.open(crop_path)))
                            break

                if images:
                    # Show preview
                    cols = st.columns(min(len(images), 10))
                    for idx, (face_id, img) in enumerate(images):
                        with cols[idx]:
                            st.image(img, caption=f"{face_id}", use_container_width=True)

                    # Button to see all faces
                    total_faces = len(faces_df[faces_df['pre_merge_cluster_id'] == merge['absorbed_cluster']])
                    if total_faces > 10:
                        st.caption(f"Showing 10 of {total_faces} faces")
                        if st.button(
                            f"🔍 View All {total_faces} Faces",
                            key=f"view_all_merge_{i}_{merge['absorbed_cluster']}"
                        ):
                            # Use the existing dialog but filter by pre_merge_cluster_id
                            show_all_pre_merge_cluster_faces(
                                merge['absorbed_cluster'],
                                total_faces,
                                faces_df,
                                crops_dir
                            )

    st.markdown("---")

    # Step 3: Show rejected merge attempts
    st.subheader(f"Step 3: Rejected Merge Attempts for Cluster {selected_cluster}")

    rejected_attempts = merge_decisions_df[
        (merge_decisions_df['action'] == 'rejected') &
        ((merge_decisions_df['cluster_a'] == selected_cluster) |
         (merge_decisions_df['cluster_b'] == selected_cluster))
    ].sort_values('iteration')

    # Deduplicate: keep only the last attempt for each unique pair
    if len(rejected_attempts) > 0:
        # Group by (cluster_a, cluster_b) and keep last iteration
        rejected_attempts['other_cluster'] = rejected_attempts.apply(
            lambda row: row['cluster_b'] if row['cluster_a'] == selected_cluster else row['cluster_a'],
            axis=1
        )
        rejected_unique = rejected_attempts.groupby('other_cluster').last().reset_index()

        st.markdown(f"**{len(rejected_unique)} unique clusters tried to merge with cluster {selected_cluster} but were rejected:**")
        st.caption(f"(Showing last rejection attempt for each pair; total {len(rejected_attempts)} rejection decisions)")

        for _, decision in rejected_unique.head(10).iterrows():
            other_cluster = decision['other_cluster']

            with st.expander(
                f"Iteration {decision['iteration']}: "
                f"Cluster {other_cluster} rejected (dist={decision['exemplar_dist']:.3f})",
                expanded=False
            ):
                st.error(f"**Rejection reason:** {decision.get('rejection_reason', 'N/A')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Exemplar Distance", f"{decision['exemplar_dist']:.3f}")
                    st.metric("Threshold", f"{decision['threshold_used']:.3f}")
                with col2:
                    st.metric("Support", f"{decision['support']}/{decision['required_support']}")

                # Show sample faces from rejected cluster
                st.markdown(f"#### Sample Faces from Cluster {other_cluster}")
                other_faces = faces_df[faces_df['cluster_id'] == other_cluster].head(5)

                if len(other_faces) > 0:
                    images = []
                    for _, face_row in other_faces.iterrows():
                        face_id = int(face_row['face_id'])  # Convert to int
                        for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
                            crop_path = crops_dir / pattern
                            if crop_path.exists():
                                images.append((face_id, Image.open(crop_path)))
                                break

                    if images:
                        cols = st.columns(min(len(images), 5))
                        for idx, (face_id, img) in enumerate(images):
                            with cols[idx]:
                                st.image(img, caption=f"{face_id}", use_container_width=True)

        if len(rejected_attempts) > 10:
            st.caption(f"Showing first 10 of {len(rejected_attempts)} rejected attempts")
    else:
        st.info(f"No rejected merge attempts for cluster {selected_cluster}")


def show_cluster_lineage_detail(
    selected_cluster: int,
    cluster_lineage: dict,
    pre_merge_clusters_df: pd.DataFrame,
    faces_df: pd.DataFrame,
    crops_dir: Path
):
    """Show detailed lineage view for a cluster."""
    source_clusters = cluster_lineage[str(selected_cluster)]

    st.markdown(f"### Post-merge Cluster {selected_cluster}")
    st.markdown(f"**Composed from {len(source_clusters)} pre-merge clusters**: {', '.join(map(str, source_clusters))}")

    # Show each source cluster
    for source_cid in source_clusters:
        st.markdown(f"#### Pre-merge Cluster {source_cid}")

        # Get cluster info
        cluster_info = pre_merge_clusters_df[pre_merge_clusters_df['cluster_id'] == source_cid]
        if len(cluster_info) > 0:
            cluster_size = cluster_info.iloc[0]['cluster_size']
            diameter = cluster_info.iloc[0]['diameter']
            st.caption(f"Size: {cluster_size} faces | Diameter: {diameter:.3f}")

        # Show faces
        source_faces = faces_df[faces_df['pre_merge_cluster_id'] == source_cid].head(10)
        if len(source_faces) > 0:
            images = []
            for _, face_row in source_faces.iterrows():
                face_id = int(face_row['face_id'])  # Convert to int
                for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
                    crop_path = crops_dir / pattern
                    if crop_path.exists():
                        images.append((face_id, Image.open(crop_path)))
                        break

            if images:
                cols = st.columns(min(len(images), 10))
                for idx, (face_id, img) in enumerate(images):
                    with cols[idx]:
                        st.image(img, caption=f"{face_id}", use_container_width=True)


def show_pre_post_comparison(
    pre_merge_clusters_df: pd.DataFrame,
    post_merge_clusters_df: pd.DataFrame,
    cluster_lineage: dict,
    faces_df: pd.DataFrame,
    crops_dir: Path
):
    """Tab 3: Compare pre-merge and post-merge clustering."""
    st.header("📊 Pre/Post Merge Comparison")

    # Overall stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pre-merge Clusters", len(pre_merge_clusters_df))
        avg_pre_diameter = pre_merge_clusters_df['diameter'].mean()
        st.metric("Avg Pre-merge Diameter", f"{avg_pre_diameter:.3f}")
    with col2:
        st.metric("Post-merge Clusters", len(post_merge_clusters_df))
        avg_post_diameter = post_merge_clusters_df['diameter'].mean()
        st.metric("Avg Post-merge Diameter", f"{avg_post_diameter:.3f}")

    # Select a post-merge cluster to inspect
    st.subheader("Inspect Cluster Lineage")

    # Only show clusters that have lineage (were affected by merging)
    merged_clusters = [int(k) for k in cluster_lineage.keys() if len(cluster_lineage[k]) > 1]

    if len(merged_clusters) > 0:
        selected_cluster = st.selectbox(
            "Select post-merge cluster (showing only merged clusters)",
            options=sorted(merged_clusters),
            format_func=lambda cid: f"Cluster {cid} (from {len(cluster_lineage[str(cid)])} pre-merge clusters)"
        )

        if selected_cluster is not None:
            show_cluster_lineage_detail(
                selected_cluster,
                cluster_lineage,
                pre_merge_clusters_df,
                faces_df,
                crops_dir
            )
    else:
        st.info("No clusters were merged (all post-merge clusters come from single pre-merge clusters)")


def show_pre_cluster_analysis(
    pre_merge_clusters_df: pd.DataFrame,
    faces_df: pd.DataFrame,
    crops_dir: Path,
    knn_graph_data: dict
):
    """Tab 4: Pre-cluster analysis with kNN diagnostics."""
    st.header("🔬 Pre-Cluster Analysis")

    st.info("""
    **What this shows**: Initial clusters created by kNN graph + connected components (BEFORE any merging).

    **Why this matters**: If pre-clusters already contain mixed people (e.g., pre-cluster 3 has Person A + Person B),
    the problem is in the initial clustering stage (transitive closure), not the merge stage.

    **What to look for**: High diameter, many outliers, or bridges (faces connecting wrong people through kNN graph).
    """)

    # Build kNN lookup for faster access
    knn_lookup = {item['face_id']: item for item in knn_graph_data['knn_neighbors']}

    # Compute outlier statistics for each cluster
    cluster_stats = []
    for _, cluster_row in pre_merge_clusters_df.iterrows():
        cluster_id = cluster_row['cluster_id']
        cluster_size = cluster_row['cluster_size']
        diameter = cluster_row['diameter']

        # Get faces in this cluster
        cluster_faces = faces_df[faces_df['pre_merge_cluster_id'] == cluster_id]

        # Compute outlier scores for all faces
        outlier_scores = []
        for _, face_row in cluster_faces.iterrows():
            face_id = int(face_row['face_id'])  # Convert to int

            if face_id in knn_lookup:
                knn_info = knn_lookup[face_id]
                neighbor_ids = knn_info['k_nearest_ids']
                neighbor_dists = knn_info['k_nearest_distances']

                neighbors_in_cluster = sum(
                    1 for nid in neighbor_ids
                    if nid in cluster_faces['face_id'].values
                )

                avg_neighbor_dist = np.mean(neighbor_dists)
                outlier_score = avg_neighbor_dist * (1 - neighbors_in_cluster / len(neighbor_ids))
                outlier_scores.append(outlier_score)

        # Compute cluster-level statistics
        avg_outlier_score = np.mean(outlier_scores) if outlier_scores else 0
        num_outliers = sum(1 for s in outlier_scores if s > 0.3)
        coherence_score = 1 - (diameter / 1.0) if diameter < 1.0 else 0  # Normalize to 0-1

        # Coherence status
        if coherence_score > 0.6:
            coherence_status = "✅ Good"
        elif coherence_score > 0.4:
            coherence_status = "⚠️ Mixed"
        else:
            coherence_status = "❌ Bad"

        cluster_stats.append({
            'Cluster ID': cluster_id,
            'Size': cluster_size,
            'Diameter': f"{diameter:.3f}",
            '# Outliers': num_outliers,
            'Avg Outlier Score': f"{avg_outlier_score:.3f}",
            'Coherence': coherence_status,
            '_diameter_raw': diameter,
            '_avg_outlier_raw': avg_outlier_score
        })

    cluster_stats_df = pd.DataFrame(cluster_stats)

    # Section 1: Summary Table
    st.subheader("📊 Pre-Cluster Summary (Initial Clustering - BEFORE Merging)")
    st.markdown("**Goal**: Identify which initial clusters contain mixed identities due to transitive closure")
    st.markdown("**Key indicators**: High diameter (>0.5) or many outliers suggest wrong faces were included in initial clustering")

    # Create display dataframe (drop raw columns for display)
    display_df = cluster_stats_df.drop(columns=['_diameter_raw', '_avg_outlier_raw'])

    # Add highlighting for problematic clusters using full dataframe for logic
    def highlight_row(row_idx):
        # Get raw values from full dataframe
        diameter = cluster_stats_df.loc[row_idx, '_diameter_raw']
        avg_outlier = cluster_stats_df.loc[row_idx, '_avg_outlier_raw']

        if diameter > 0.5 or avg_outlier > 0.25:
            return ['background-color: #ffcccc'] * len(display_df.columns)
        else:
            return [''] * len(display_df.columns)

    # Apply highlighting
    styled_df = display_df.style.apply(lambda row: highlight_row(row.name), axis=1)

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

    st.markdown("---")

    # Section 2: Detailed Analysis
    st.subheader("🔍 Detailed Pre-Cluster Face Analysis")
    st.markdown("**Analyzing**: Initial clusters (before any merging happened)")

    # Select cluster to analyze
    selected_cluster = st.selectbox(
        "Select PRE-MERGE cluster to analyze",
        options=sorted(pre_merge_clusters_df['cluster_id'].tolist()),
        format_func=lambda cid: f"Pre-cluster {cid} (Size: {cluster_stats_df[cluster_stats_df['Cluster ID']==cid].iloc[0]['Size']})"
    )

    # === METRIC EXPLANATIONS ===
    with st.expander("❓ Metric Explanations", expanded=False):
        st.markdown("""
        ### Understanding the Metrics

        **Outlier Score** = `avg_neighbor_dist × (1 - neighbors_in_cluster / K)`
        - Combines distance and connectivity
        - High score (>0.3) = face is far from neighbors AND most neighbors are outside cluster
        - Range: 0 (core member) to ~0.5+ (strong outlier)

        **Bridge %** = `(neighbors_outside_cluster / K) × 100`
        - Percentage of K=5 nearest neighbors that are in DIFFERENT pre-clusters
        - 0% = All neighbors in same cluster (strong belonging)
        - 100% = All neighbors in other clusters (transitive bridge!)
        - High bridge % indicates face connects multiple clusters through kNN graph

        **Neighbors In/Out**
        - Of the K=5 nearest neighbors (by embedding distance):
          - **In**: How many are in the same pre-cluster
          - **Out**: How many are in different pre-clusters
        - Example: "In: 2 | Out: 3" means 2 neighbors share this cluster, 3 are elsewhere

        **Distance to Exemplar**
        - Cosine distance to nearest exemplar face
        - Low (<0.3) = Face is similar to cluster's representative
        - High (>0.5) = Face doesn't match cluster core (possible wrong inclusion)

        **Distance to Centroid**
        - Cosine distance to cluster's average embedding
        - Measures how "typical" this face is for the cluster

        **Closest External Cluster**
        - Which other pre-cluster has the closest face to this one
        - Distance shown in parentheses
        - 🎯 appears if distance < 0.35 (very close - possible misassignment)

        **Diameter**
        - Maximum pairwise distance between ANY two faces in cluster
        - High diameter (>0.5) suggests cluster contains different people
        - The face pair creating this max distance often reveals the problem
        """)

    if selected_cluster is not None:
        cluster_info = pre_merge_clusters_df[pre_merge_clusters_df['cluster_id'] == selected_cluster].iloc[0]
        cluster_faces = faces_df[faces_df['pre_merge_cluster_id'] == selected_cluster]

        # Load embeddings for distance computation
        has_embeddings = False
        embedding_lookup = {}

        try:
            # Get embeddings source from export directory (already set from user input)
            export_summary_path = export_dir / 'export_summary.json'

            st.info(f"🔍 Debug: Export dir = {export_dir}")
            st.info(f"🔍 Debug: Looking for export_summary.json at: {export_summary_path}")

            if export_summary_path.exists():
                with open(export_summary_path) as f:
                    export_summary = json.load(f)
                    embeddings_rel_path = export_summary['embeddings_source']

                    st.info(f"🔍 Debug: Embeddings source from JSON: {embeddings_rel_path}")

                    # Try to resolve path (could be relative or absolute)
                    embeddings_path = Path(embeddings_rel_path)

                    # If relative, make it relative to project root (parent of results/)
                    if not embeddings_path.is_absolute():
                        # Try 1: Relative to project root
                        project_root = export_dir.parent.parent  # Go up from export_dir to results/ to root
                        embeddings_path = project_root / embeddings_rel_path
                        st.info(f"🔍 Debug: Try 1 (relative to project root): {embeddings_path}")

                    if not embeddings_path.exists():
                        # Try 2: Relative to export directory's parent
                        embeddings_path = export_dir.parent / Path(embeddings_rel_path).name
                        st.info(f"🔍 Debug: Try 2 (same dir as export): {embeddings_path}")

                    if not embeddings_path.exists():
                        # Try 3: Parse the path and use just the filename in expected location
                        # embeddings_rel_path might be like "results\\Google_Germany\\embeddings_*.npy"
                        filename = Path(embeddings_rel_path).name
                        embeddings_path = export_dir.parent / filename
                        st.info(f"🔍 Debug: Try 3 (filename only): {embeddings_path}")

                    if embeddings_path.exists():
                        st.success(f"✅ Found embeddings at: {embeddings_path}")
                        # Load embeddings array (plain numpy array, not dict)
                        all_embeddings = np.load(embeddings_path, allow_pickle=True)

                        # Handle both formats: plain array or dict
                        if isinstance(all_embeddings, np.ndarray) and all_embeddings.dtype != object:
                            # Plain array format: match by index
                            # Get face IDs from faces_df (should be in same order as embeddings)
                            all_face_ids = faces_df['face_id'].tolist()
                            embedding_lookup = {fid: all_embeddings[i] for i, fid in enumerate(all_face_ids) if i < len(all_embeddings)}
                        else:
                            # Dict format (legacy)
                            embeddings_data = all_embeddings.item()
                            all_embeddings = embeddings_data['embeddings']
                            all_face_ids = [int(fid) for fid in embeddings_data['face_ids']]
                            embedding_lookup = {fid: all_embeddings[i] for i, fid in enumerate(all_face_ids)}

                        has_embeddings = True
                        st.success(f"✅ Loaded {len(all_embeddings)} embeddings successfully!")
                    else:
                        st.error(f"❌ Embeddings file not found after trying 3 paths")
                        st.error(f"Original path from JSON: {embeddings_rel_path}")
                        st.error(f"Last attempted path: {embeddings_path}")
            else:
                st.error(f"❌ export_summary.json not found at: {export_summary_path}")
        except Exception as e:
            st.error(f"❌ Failed to load embeddings: {e}")
            import traceback
            st.code(traceback.format_exc())

        # Get exemplar IDs
        exemplar_ids_str = cluster_info['exemplar_ids']
        exemplar_ids = [int(x) for x in exemplar_ids_str.split(',') if x]

        # === SECTION A: CLUSTER OVERVIEW ===
        st.subheader(f"📊 Pre-cluster {selected_cluster} Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Size", len(cluster_faces))
        with col2:
            st.metric("Diameter", f"{cluster_info['diameter']:.3f}",
                     help="Max distance between any two faces in cluster")
        with col3:
            st.metric("# Exemplars", len(exemplar_ids))
        with col4:
            if st.button(f"👁️ View All Faces", use_container_width=True):
                show_all_pre_merge_cluster_faces(selected_cluster, faces_df, crops_dir)

        st.markdown("---")

        # === SECTION B: EXEMPLARS ===
        st.subheader("⭐ Cluster Exemplars")
        st.caption("Exemplars are representative faces selected to represent this cluster's core identity")

        if exemplar_ids:
            exemplar_cols = st.columns(min(len(exemplar_ids), 6))
            for i, ex_id in enumerate(exemplar_ids[:6]):  # Show max 6
                with exemplar_cols[i % 6]:
                    # Load exemplar image
                    ex_img = None
                    for pattern in [f"face_{ex_id:04d}_aligned.jpg", f"face_{ex_id:04d}.jpg"]:
                        crop_path = crops_dir / pattern
                        if crop_path.exists():
                            ex_img = Image.open(crop_path)
                            ex_img = ex_img.resize((120, 120))
                            break
                    if ex_img:
                        st.image(ex_img, use_container_width=False)
                    st.caption(f"Exemplar {ex_id}")
        else:
            st.info("No exemplars found")

        # Exemplar Distance Matrix
        if has_embeddings and exemplar_ids and len(exemplar_ids) > 1:
            st.markdown("**Exemplar Distance Matrix**")
            st.caption("Distances between exemplars - high distances suggest mixed cluster")

            # Compute pairwise distances between exemplars
            exemplar_dist_matrix = []
            for i, ex_id1 in enumerate(exemplar_ids):
                row = []
                for j, ex_id2 in enumerate(exemplar_ids):
                    if ex_id1 in embedding_lookup and ex_id2 in embedding_lookup:
                        emb1 = embedding_lookup[ex_id1]
                        emb2 = embedding_lookup[ex_id2]
                        dist = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        row.append(f"{dist:.3f}")
                    else:
                        row.append("N/A")
                exemplar_dist_matrix.append(row)

            # Display as dataframe
            exemplar_df = pd.DataFrame(
                exemplar_dist_matrix,
                index=[f"Ex {eid}" for eid in exemplar_ids],
                columns=[f"Ex {eid}" for eid in exemplar_ids]
            )
            st.dataframe(exemplar_df, use_container_width=True)

            # Analyze exemplar quality
            max_exemplar_dist = 0
            for i, ex_id1 in enumerate(exemplar_ids):
                for j, ex_id2 in enumerate(exemplar_ids[i+1:], i+1):
                    if ex_id1 in embedding_lookup and ex_id2 in embedding_lookup:
                        emb1 = embedding_lookup[ex_id1]
                        emb2 = embedding_lookup[ex_id2]
                        dist = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        max_exemplar_dist = max(max_exemplar_dist, dist)

            if max_exemplar_dist > 0.5:
                st.error(f"🚨 Exemplars are very far apart (max={max_exemplar_dist:.3f})! Cluster likely contains different people.")
            elif max_exemplar_dist > 0.4:
                st.warning(f"⚠️ Exemplars are moderately distant (max={max_exemplar_dist:.3f}). Cluster may be mixed.")

        st.markdown("---")

        # === UMAP VISUALIZATION ===
        if has_embeddings and len(cluster_faces) >= 5:
            st.subheader("🗺️ UMAP Visualization")
            st.caption("2D projection of cluster faces - distinct groups suggest mixed identities")

            try:
                from umap import UMAP

                # Get embeddings for all faces in cluster
                cluster_face_ids = cluster_faces['face_id'].tolist()
                cluster_embeddings = []
                cluster_fids_with_emb = []

                for fid in cluster_face_ids:
                    fid_int = int(fid)
                    if fid_int in embedding_lookup:
                        cluster_embeddings.append(embedding_lookup[fid_int])
                        cluster_fids_with_emb.append(fid_int)

                if len(cluster_embeddings) >= 5:
                    # Run UMAP
                    reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(cluster_embeddings)-1))
                    umap_coords = reducer.fit_transform(np.array(cluster_embeddings))

                    # Create scatter plot
                    import plotly.graph_objects as go

                    # Color by outlier score
                    colors = []
                    hover_texts = []
                    for fid in cluster_fids_with_emb:
                        face_metrics = face_analysis_df[face_analysis_df['face_id'] == fid]
                        if len(face_metrics) > 0:
                            outlier = face_metrics.iloc[0]['outlier_score']
                            colors.append(outlier)
                            hover_texts.append(f"Face {fid}<br>Outlier: {outlier:.3f}")
                        else:
                            colors.append(0)
                            hover_texts.append(f"Face {fid}")

                    fig = go.Figure()

                    # Add all points
                    fig.add_trace(go.Scatter(
                        x=umap_coords[:, 0],
                        y=umap_coords[:, 1],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=colors,
                            colorscale='Reds',
                            colorbar=dict(title="Outlier<br>Score"),
                            line=dict(width=1, color='white')
                        ),
                        text=hover_texts,
                        hovertemplate='%{text}<extra></extra>'
                    ))

                    # Highlight exemplars
                    if exemplar_ids:
                        exemplar_coords = []
                        exemplar_labels = []
                        for ex_id in exemplar_ids:
                            if ex_id in cluster_fids_with_emb:
                                idx = cluster_fids_with_emb.index(ex_id)
                                exemplar_coords.append(umap_coords[idx])
                                exemplar_labels.append(f"Exemplar {ex_id}")

                        if exemplar_coords:
                            exemplar_coords = np.array(exemplar_coords)
                            fig.add_trace(go.Scatter(
                                x=exemplar_coords[:, 0],
                                y=exemplar_coords[:, 1],
                                mode='markers+text',
                                marker=dict(
                                    size=15,
                                    color='gold',
                                    symbol='star',
                                    line=dict(width=2, color='black')
                                ),
                                text=["⭐"] * len(exemplar_coords),
                                textposition="top center",
                                hovertext=exemplar_labels,
                                hovertemplate='%{hovertext}<extra></extra>',
                                name='Exemplars'
                            ))

                    fig.update_layout(
                        title=f"UMAP Projection of Pre-cluster {selected_cluster}",
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        showlegend=False,
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("⭐ = Exemplars | Color = Outlier score (darker red = higher outlier)")
                    st.caption("Look for: Distinct clusters (multiple groups), outliers far from main group, exemplars not in center")

            except ImportError:
                st.info("Install UMAP for visualization: `pip install umap-learn plotly`")
            except Exception as e:
                st.warning(f"UMAP visualization failed: {e}")

        st.markdown("---")

        # === SECTION C: DIAMETER ANALYSIS ===
        if has_embeddings:
            st.subheader("📏 Diameter Analysis")
            st.caption("Which face pair creates the maximum distance (diameter)?")

            # Compute pairwise distances for all faces in cluster
            cluster_face_ids = cluster_faces['face_id'].tolist()
            max_dist = 0
            max_pair = (None, None)

            for i, fid1 in enumerate(cluster_face_ids):
                fid1_int = int(fid1)
                if fid1_int not in embedding_lookup:
                    continue
                emb1 = embedding_lookup[fid1_int]

                for fid2 in cluster_face_ids[i+1:]:
                    fid2_int = int(fid2)
                    if fid2_int not in embedding_lookup:
                        continue
                    emb2 = embedding_lookup[fid2_int]

                    # Cosine distance
                    dist = 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    if dist > max_dist:
                        max_dist = dist
                        max_pair = (fid1_int, fid2_int)

            if max_pair[0] is not None:
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    # Load first face
                    for pattern in [f"face_{max_pair[0]:04d}_aligned.jpg", f"face_{max_pair[0]:04d}.jpg"]:
                        crop_path = crops_dir / pattern
                        if crop_path.exists():
                            img = Image.open(crop_path)
                            st.image(img, use_container_width=True)
                            break
                    st.caption(f"Face {max_pair[0]}")

                with col2:
                    st.metric("Max Distance", f"{max_dist:.3f}")
                    st.caption("⬅️ These two faces ➡️")
                    if max_dist > 0.5:
                        st.error("🚨 Very high! Likely different people")

                with col3:
                    # Load second face
                    for pattern in [f"face_{max_pair[1]:04d}_aligned.jpg", f"face_{max_pair[1]:04d}.jpg"]:
                        crop_path = crops_dir / pattern
                        if crop_path.exists():
                            img = Image.open(crop_path)
                            st.image(img, use_container_width=True)
                            break
                    st.caption(f"Face {max_pair[1]}")

            st.markdown("---")

        # === SECTION D: FACE-LEVEL ANALYSIS ===
        st.subheader("🔬 Face-Level Diagnostics")
        st.caption("Detailed metrics for each face in this pre-cluster")

        # Compute enhanced face-level metrics
        face_analysis = []
        for _, face_row in cluster_faces.iterrows():
            face_id = int(face_row['face_id'])

            # Initialize metrics
            metrics = {
                'face_id': face_id,
                'image_path': face_row.get('image_path', 'Unknown'),
                'neighbors_in_cluster': 0,
                'neighbors_outside': 0,
                'avg_neighbor_dist': 0,
                'outlier_score': 0,
                'dist_to_nearest_exemplar': None,
                'dist_to_centroid': None,
                'closest_external_cluster': None,
                'closest_external_dist': None,
                'bridge_score': 0
            }

            # kNN-based metrics
            if face_id in knn_lookup:
                knn_info = knn_lookup[face_id]
                neighbor_ids = knn_info['k_nearest_ids']
                neighbor_dists = knn_info['k_nearest_distances']

                neighbors_in_cluster = sum(
                    1 for nid in neighbor_ids
                    if nid in cluster_faces['face_id'].values
                )

                avg_neighbor_dist = np.mean(neighbor_dists)
                outlier_score = avg_neighbor_dist * (1 - neighbors_in_cluster / len(neighbor_ids))

                metrics['neighbors_in_cluster'] = neighbors_in_cluster
                metrics['neighbors_outside'] = len(neighbor_ids) - neighbors_in_cluster
                metrics['avg_neighbor_dist'] = avg_neighbor_dist
                metrics['outlier_score'] = outlier_score

                # Bridge score: how many neighbors are outside cluster
                metrics['bridge_score'] = metrics['neighbors_outside'] / len(neighbor_ids)

            # Embedding-based metrics
            if has_embeddings and face_id in embedding_lookup:
                face_emb = embedding_lookup[face_id]

                # Distance to each exemplar in THIS cluster (with IDs)
                exemplar_dists_with_ids = []
                if exemplar_ids:
                    for ex_id in exemplar_ids:
                        if ex_id in embedding_lookup:
                            ex_emb = embedding_lookup[ex_id]
                            dist = 1 - np.dot(face_emb, ex_emb) / (np.linalg.norm(face_emb) * np.linalg.norm(ex_emb))
                            exemplar_dists_with_ids.append((ex_id, dist))

                    if exemplar_dists_with_ids:
                        # Sort by distance
                        exemplar_dists_with_ids.sort(key=lambda x: x[1])
                        metrics['nearest_exemplar_id'] = exemplar_dists_with_ids[0][0]
                        metrics['dist_to_nearest_exemplar'] = exemplar_dists_with_ids[0][1]
                        metrics['all_exemplar_dists'] = exemplar_dists_with_ids  # Store all for display

                # Distance to cluster centroid (average of all faces)
                cluster_embs = []
                for cid in cluster_face_ids:
                    cid_int = int(cid)
                    if cid_int in embedding_lookup:
                        cluster_embs.append(embedding_lookup[cid_int])
                if cluster_embs:
                    centroid = np.mean(cluster_embs, axis=0)
                    metrics['dist_to_centroid'] = 1 - np.dot(face_emb, centroid) / (np.linalg.norm(face_emb) * np.linalg.norm(centroid))

                # Closest EXEMPLAR from DIFFERENT clusters (not just any face)
                min_external_exemplar_dist = float('inf')
                closest_external_cluster = None
                closest_external_exemplar_id = None

                for _, other_cluster_row in pre_merge_clusters_df.iterrows():
                    other_cid = other_cluster_row['cluster_id']
                    if other_cid == selected_cluster:
                        continue

                    # Get exemplars of this other cluster
                    other_exemplar_ids_str = other_cluster_row['exemplar_ids']
                    other_exemplar_ids = [int(x) for x in other_exemplar_ids_str.split(',') if x]

                    # Find closest exemplar from this other cluster
                    for other_ex_id in other_exemplar_ids:
                        if other_ex_id in embedding_lookup:
                            other_ex_emb = embedding_lookup[other_ex_id]
                            dist = 1 - np.dot(face_emb, other_ex_emb) / (np.linalg.norm(face_emb) * np.linalg.norm(other_ex_emb))
                            if dist < min_external_exemplar_dist:
                                min_external_exemplar_dist = dist
                                closest_external_cluster = other_cid
                                closest_external_exemplar_id = other_ex_id

                if closest_external_cluster is not None:
                    metrics['closest_external_cluster'] = closest_external_cluster
                    metrics['closest_external_exemplar_id'] = closest_external_exemplar_id
                    metrics['closest_external_dist'] = min_external_exemplar_dist

            face_analysis.append(metrics)

        face_analysis_df = pd.DataFrame(face_analysis).sort_values('outlier_score', ascending=False)

        # Display face table with enhanced metrics
        for idx, row in face_analysis_df.iterrows():
            face_id = row['face_id']
            outlier_score = row['outlier_score']

            # Determine status
            if outlier_score > 0.3:
                status = "❌ Outlier"
            elif outlier_score > 0.2:
                status = "⚠️ Boundary"
            else:
                status = "✅ Core"

            # Load face image
            face_img = None
            for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
                crop_path = crops_dir / pattern
                if crop_path.exists():
                    face_img = Image.open(crop_path)
                    face_img = face_img.resize((80, 80))
                    break

            # Create expandable row for each face
            with st.expander(f"**Face {face_id}** {status} | Outlier: {outlier_score:.3f}", expanded=False):
                cols = st.columns([1, 3])

                with cols[0]:
                    if face_img:
                        st.image(face_img, use_container_width=True)

                with cols[1]:
                    # Show distances to THIS cluster's exemplars
                    st.markdown("**📍 This Cluster's Exemplars:**")
                    if 'all_exemplar_dists' in row and row['all_exemplar_dists']:
                        for ex_id, dist in row['all_exemplar_dists']:
                            st.caption(f"  • Exemplar {ex_id}: {dist:.3f}")
                    elif row['dist_to_nearest_exemplar'] is not None:
                        st.caption(f"  • Nearest: {row['dist_to_nearest_exemplar']:.3f}")
                    else:
                        st.caption("  • N/A")

                    st.markdown("---")

                    # Show distance to CLOSEST EXEMPLAR from DIFFERENT cluster
                    st.markdown("**🎯 Closest External Exemplar:**")
                    if row['closest_external_cluster'] is not None:
                        ext_cluster = row['closest_external_cluster']
                        ext_exemplar = row.get('closest_external_exemplar_id', 'Unknown')
                        ext_dist = row['closest_external_dist']
                        own_dist = row['dist_to_nearest_exemplar']

                        # Highlight if closer to external exemplar than own cluster
                        if own_dist and ext_dist < own_dist:
                            st.error(f"🚨 **Pre-cluster {ext_cluster}, Exemplar {ext_exemplar}: {ext_dist:.3f}**")
                            st.caption(f"⚠️ CLOSER than own cluster ({own_dist:.3f})! Face may belong in cluster {ext_cluster}")
                        else:
                            st.success(f"✓ Pre-cluster {ext_cluster}, Exemplar {ext_exemplar}: {ext_dist:.3f}")
                            st.caption(f"Further than own cluster ({own_dist:.3f})")
                    else:
                        st.caption("  • N/A")

                    st.markdown("---")

                    # kNN diagnostics
                    st.markdown("**🔗 kNN Graph Connectivity:**")
                    st.caption(f"  • Neighbors in cluster: {row['neighbors_in_cluster']}")
                    st.caption(f"  • Neighbors outside: {row['neighbors_outside']}")
                    bridge_pct = int(row['bridge_score'] * 100)
                    st.caption(f"  • Bridge score: {bridge_pct}% ({'High - likely bridge' if bridge_pct > 50 else 'OK'})")

            # Show KNN neighbors in expandable section
            if face_id in knn_lookup:
                with st.expander(f"🔗 K-Nearest Neighbors for Face {face_id}", expanded=False):
                    knn_info = knn_lookup[face_id]
                    neighbor_ids = knn_info['k_nearest_ids']
                    neighbor_dists = knn_info['k_nearest_distances']

                    st.caption("These are the 5 closest faces by embedding distance (used to build kNN graph)")

                    # Display neighbors in a grid
                    neighbor_cols = st.columns(5)
                    for i, (nid, dist) in enumerate(zip(neighbor_ids, neighbor_dists)):
                        nid_int = int(nid)

                        # Check if neighbor is in same cluster
                        neighbor_in_cluster = nid in cluster_faces['face_id'].values

                        # Get neighbor's cluster
                        neighbor_cluster_id = None
                        neighbor_row = faces_df[faces_df['face_id'] == nid]
                        if len(neighbor_row) > 0:
                            neighbor_cluster_id = int(neighbor_row.iloc[0]['pre_merge_cluster_id'])

                        with neighbor_cols[i]:
                            # Load neighbor image
                            neighbor_img = None
                            for pattern in [f"face_{nid_int:04d}_aligned.jpg", f"face_{nid_int:04d}.jpg"]:
                                crop_path = crops_dir / pattern
                                if crop_path.exists():
                                    neighbor_img = Image.open(crop_path)
                                    neighbor_img = neighbor_img.resize((80, 80))
                                    break

                            if neighbor_img:
                                st.image(neighbor_img, use_container_width=False)

                            st.caption(f"**Face {nid_int}**")
                            st.caption(f"Dist: {dist:.3f}")

                            if neighbor_in_cluster:
                                st.success(f"✅ Same C{selected_cluster}")
                            else:
                                if neighbor_cluster_id is not None:
                                    st.error(f"❌ Pre-C{neighbor_cluster_id}")
                                else:
                                    st.warning("❓ Unknown")

            st.markdown("---")


@st.dialog("📋 View All Faces in Pre-Cluster", width="large")
def show_all_pre_merge_cluster_faces(cluster_id: int, faces_df: pd.DataFrame, crops_dir: Path):
    """Display all faces in a pre-merge cluster."""
    st.subheader(f"All Faces in Pre-cluster {cluster_id}")

    cluster_faces = faces_df[faces_df['pre_merge_cluster_id'] == cluster_id]

    st.caption(f"Total faces: {len(cluster_faces)}")

    # Display in grid
    cols_per_row = 6
    for i in range(0, len(cluster_faces), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, face_row) in enumerate(cluster_faces.iloc[i:i+cols_per_row].iterrows()):
            face_id = int(face_row['face_id'])  # Convert to int

            # Try to load image
            face_img = None
            for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
                crop_path = crops_dir / pattern
                if crop_path.exists():
                    face_img = Image.open(crop_path)
                    # Resize to uniform size
                    face_img = face_img.resize((120, 120))
                    break

            with cols[j]:
                if face_img:
                    st.image(face_img, use_container_width=False)
                st.caption(f"Face {face_id}")


@st.dialog("Face Deep Dive", width="large")
def show_face_deep_dive(
    face_id: int,
    current_cluster: int,
    faces_df: pd.DataFrame,
    pre_merge_clusters_df: pd.DataFrame,
    crops_dir: Path,
    knn_graph_data: dict
):
    """Deep dive modal for a specific face."""
    st.subheader(f"Deep Dive: Face {face_id}")

    # Ensure face_id is int
    face_id = int(face_id)

    # Get face info
    face_row = faces_df[faces_df['face_id'] == face_id].iloc[0]
    image_path = face_row.get('image_path', 'Unknown')

    st.caption(f"**Image**: {image_path}")
    st.caption(f"**Current cluster**: Pre-cluster {current_cluster}")

    # Load face image
    face_img = None
    for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
        crop_path = crops_dir / pattern
        if crop_path.exists():
            face_img = Image.open(crop_path)
            break

    if face_img:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(face_img, caption=f"Face {face_id}", use_container_width=True)

    st.markdown("---")

    # Section A: K-Nearest Neighbors
    st.markdown("### A. K-Nearest Neighbors")
    st.caption("These are the closest faces in the dataset (why this face connected to the cluster)")

    knn_lookup = {item['face_id']: item for item in knn_graph_data['knn_neighbors']}

    if face_id not in knn_lookup:
        st.warning(f"No kNN data found for face {face_id}")
        return

    knn_info = knn_lookup[face_id]
    neighbor_ids = knn_info['k_nearest_ids']
    neighbor_dists = knn_info['k_nearest_distances']

    # Show neighbors in table with images
    for i, (nid, dist) in enumerate(zip(neighbor_ids, neighbor_dists), 1):
        nid = int(nid)  # Convert to int
        neighbor_row = faces_df[faces_df['face_id'] == nid]
        if len(neighbor_row) == 0:
            continue

        neighbor_row = neighbor_row.iloc[0]
        neighbor_cluster = int(neighbor_row['pre_merge_cluster_id'])

        in_cluster = neighbor_cluster == current_cluster
        status = "✅ Same cluster" if in_cluster else f"❌ Different (C{neighbor_cluster})"

        # Load neighbor image
        neighbor_img = None
        for pattern in [f"face_{nid:04d}_aligned.jpg", f"face_{nid:04d}.jpg"]:
            crop_path = crops_dir / pattern
            if crop_path.exists():
                neighbor_img = Image.open(crop_path)
                break

        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            if neighbor_img:
                st.image(neighbor_img, use_container_width=True)

        with col2:
            st.markdown(f"**Neighbor #{i}: Face {nid}**")
            st.caption(status)

        with col3:
            st.metric("Distance", f"{dist:.3f}")

        if i < len(neighbor_ids):
            st.markdown("---")

    # Section B: Alternative Cluster Assignments
    st.markdown("### B. Alternative Cluster Assignments")
    st.caption("Which other clusters is this face close to?")

    # For each cluster, find minimum distance to any face in that cluster
    alternative_assignments = []

    for _, cluster_info in pre_merge_clusters_df.iterrows():
        cluster_id = cluster_info['cluster_id']

        if cluster_id == current_cluster:
            continue  # Skip current cluster

        # Get faces in this cluster
        cluster_face_ids = faces_df[faces_df['pre_merge_cluster_id'] == cluster_id]['face_id'].tolist()

        # Find minimum distance to any face in this cluster
        min_dist = float('inf')
        closest_face = None

        for nid, dist in zip(neighbor_ids, neighbor_dists):
            if nid in cluster_face_ids and dist < min_dist:
                min_dist = dist
                closest_face = nid

        if closest_face is not None:
            alternative_assignments.append({
                'cluster_id': cluster_id,
                'min_distance': min_dist,
                'closest_face': closest_face,
                'cluster_size': cluster_info['cluster_size']
            })

    # Sort by distance
    alternative_assignments.sort(key=lambda x: x['min_distance'])

    # Show top 5 alternatives
    st.markdown("**Top 5 alternative clusters:**")

    for alt in alternative_assignments[:5]:
        cluster_id = alt['cluster_id']
        min_dist = alt['min_distance']
        comparison = "🎯 Closer!" if min_dist < neighbor_dists[0] else "Farther"

        st.markdown(f"- **Pre-cluster {cluster_id}**: min_dist={min_dist:.3f} ({comparison})")


def save_labels(export_dir: Path, labels_df: pd.DataFrame):
    """Save corrected labels to CSV."""
    labels_path = export_dir / 'corrected_labels.csv'
    labels_df.to_csv(labels_path, index=False)

    # Also save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_clusters': len(labels_df),
        'n_identities': labels_df['corrected_identity'].nunique(),
        'identity_counts': labels_df['corrected_identity'].value_counts().to_dict()
    }

    metadata_path = export_dir / 'labeling_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return labels_path, metadata_path


# Sidebar: Load data
st.sidebar.header("📂 Data Source")

default_dir = "results/face_clustering_training/test_v2"
export_dir_input = st.sidebar.text_input(
    "Export directory",
    value=default_dir,
    help="Directory containing faces.csv, clusters.csv from export script"
)

if st.sidebar.button("Load Data") or st.session_state.export_dir is None:
    export_dir = Path(export_dir_input)

    if not export_dir.exists():
        st.error(f"Directory not found: {export_dir}")
    elif not (export_dir / 'faces.csv').exists():
        st.error(f"faces.csv not found in {export_dir}")
    elif not (export_dir / 'clusters.csv').exists():
        st.error(f"clusters.csv not found in {export_dir}")
    else:
        st.session_state.export_dir = export_dir
        st.session_state.faces_df, st.session_state.clusters_df, st.session_state.labels_df = load_csvs(export_dir)

        # Extract identity suggestions from existing labels
        st.session_state.identity_suggestions = sorted(
            st.session_state.labels_df['corrected_identity'].unique().tolist()
        )

        st.sidebar.success(f"✅ Loaded {len(st.session_state.clusters_df)} clusters")

# Main UI
if st.session_state.export_dir is not None:
    export_dir = st.session_state.export_dir
    faces_df = st.session_state.faces_df
    clusters_df = st.session_state.clusters_df
    labels_df = st.session_state.labels_df

    # Check for face crops directory
    crops_dir = None
    checked_paths = []

    # FIRST: Check export_summary.json for embeddings source path (most reliable)
    summary_file = export_dir / 'export_summary.json'
    if summary_file.exists():
        try:
            import json
            with open(summary_file) as f:
                summary = json.load(f)

            # New format: embeddings_dir points to directory with face_crops
            if 'embeddings_dir' in summary:
                embeddings_dir = Path(summary['embeddings_dir'])
                crops_from_summary = embeddings_dir / 'face_crops'
                checked_paths.append(f"{crops_from_summary} (from export_summary.json)")
                if crops_from_summary.exists():
                    crops_dir = crops_from_summary
        except Exception as e:
            st.warning(f"Could not read export_summary.json: {e}")

    # Fallback: Try common locations
    if crops_dir is None:
        # 1. In export directory itself
        crops_in_export = export_dir / 'face_crops'
        checked_paths.append(str(crops_in_export))
        if crops_in_export.exists():
            crops_dir = crops_in_export

    if crops_dir is None:
        # 2. In parent directory
        crops_in_parent = export_dir.parent / 'face_crops'
        checked_paths.append(str(crops_in_parent))
        if crops_in_parent.exists():
            crops_dir = crops_in_parent

    # Display results
    if crops_dir is None or not crops_dir.exists():
        st.warning(f"⚠️ Face crops directory not found. Checked:")
        for path in checked_paths:
            st.markdown(f"- `{path}`")
        st.error("**Solution:** Re-export with the latest script version to save embeddings_dir in export_summary.json")
        st.code("""python scripts/export_clustering_data.py \\
    --embeddings results/Budapest2025_Google/embeddings_*.npy \\
    --output results/training/Budapest2025_Google""", language="bash")
        st.info("Face images won't be displayed, but you can still assign labels based on cluster IDs.")
        crops_dir = export_dir / 'face_crops'  # Set dummy path to prevent errors
    else:
        st.sidebar.success(f"✅ Face crops: {crops_dir}")

    # Load diagnostic data if available
    merge_decisions_path = export_dir / 'merge_decisions.csv'
    pre_merge_clusters_path = export_dir / 'pre_merge_clusters.csv'
    cluster_lineage_path = export_dir / 'cluster_lineage.json'
    knn_graph_path = export_dir / 'knn_graph.json'

    has_diagnostics = (
        merge_decisions_path.exists() and
        pre_merge_clusters_path.exists() and
        cluster_lineage_path.exists() and
        'pre_merge_cluster_id' in faces_df.columns  # Check column exists
    )

    has_knn_graph = knn_graph_path.exists()

    # Create tabs
    if has_diagnostics:
        # Load diagnostic data
        merge_decisions_df = pd.read_csv(merge_decisions_path)
        pre_merge_clusters_df = pd.read_csv(pre_merge_clusters_path)
        with open(cluster_lineage_path) as f:
            cluster_lineage = json.load(f)

        # Load kNN graph if available
        knn_graph_data = None
        if has_knn_graph:
            with open(knn_graph_path) as f:
                knn_graph_data = json.load(f)

        # Create tabs (add Pre-Cluster Analysis if kNN data available)
        if has_knn_graph:
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏷️ Label Clusters",
                "🔍 Merge Analysis",
                "📊 Pre/Post Comparison",
                "🔬 Pre-Cluster Analysis"
            ])
        else:
            tab1, tab2, tab3 = st.tabs([
                "🏷️ Label Clusters",
                "🔍 Merge Analysis",
                "📊 Pre/Post Comparison"
            ])
            tab4 = None
    else:
        st.warning("⚠️ Diagnostic data not found or incomplete. Re-export with latest script to enable merge analysis.")

        # Show what's missing
        missing = []
        if not merge_decisions_path.exists():
            missing.append("merge_decisions.csv")
        if not pre_merge_clusters_path.exists():
            missing.append("pre_merge_clusters.csv")
        if not cluster_lineage_path.exists():
            missing.append("cluster_lineage.json")
        if 'pre_merge_cluster_id' not in faces_df.columns:
            missing.append("pre_merge_cluster_id column in faces.csv")

        if missing:
            st.error(f"**Missing**: {', '.join(missing)}")

        st.info("Run: `python scripts/export_clustering_data.py --embeddings <path> --output <this_directory>`")
        tab1 = st.container()
        tab2 = None
        tab3 = None
        merge_decisions_df = None
        pre_merge_clusters_df = None
        cluster_lineage = None

    # TAB 1: Main labeling interface
    with tab1:
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Faces", len(faces_df))
    with col2:
        st.metric("Clusters", len(clusters_df))
    with col3:
        n_identities = labels_df['corrected_identity'].nunique()
        st.metric("Unique Identities", n_identities)
    with col4:
        n_noise = len(faces_df[faces_df['cluster_id'] == -1])
        st.metric("Noise Faces", n_noise)

    # Identity distribution
    with st.expander("📊 Identity Distribution", expanded=False):
        identity_counts = labels_df['corrected_identity'].value_counts()
        st.bar_chart(identity_counts)
        st.dataframe(
            identity_counts.reset_index().rename(columns={'index': 'Identity', 'corrected_identity': 'Count'}),
            hide_index=True
        )

    # Labeling interface
    st.header("🏷️ Assign Identities to Clusters")

    # Sort options
    sort_by = st.selectbox(
        "Sort clusters by",
        ["Cluster ID", "Size (largest first)", "Size (smallest first)"],
        index=0
    )

    if sort_by == "Cluster ID":
        sorted_clusters = clusters_df.sort_values('cluster_id')
    elif sort_by == "Size (largest first)":
        sorted_clusters = clusters_df.sort_values('cluster_size', ascending=False)
    else:
        sorted_clusters = clusters_df.sort_values('cluster_size', ascending=True)

    # Pagination
    clusters_per_page = st.slider("Clusters per page", min_value=5, max_value=50, value=10, step=5)
    n_pages = (len(sorted_clusters) + clusters_per_page - 1) // clusters_per_page
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)

    start_idx = (page - 1) * clusters_per_page
    end_idx = min(start_idx + clusters_per_page, len(sorted_clusters))
    page_clusters = sorted_clusters.iloc[start_idx:end_idx]

    st.markdown(f"Showing clusters {start_idx + 1}-{end_idx} of {len(sorted_clusters)}")

    # Track changes
    changes_made = False

    # Display each cluster
    for idx, cluster_row in page_clusters.iterrows():
        cluster_id = cluster_row['cluster_id']
        cluster_size = cluster_row['cluster_size']

        st.markdown("---")

        # Cluster header
        col_header, col_expand, col_identity = st.columns([2, 1, 3])

        with col_header:
            st.subheader(f"Cluster {cluster_id}")
            st.caption(f"Size: {cluster_size} faces | Diameter: {cluster_row['diameter']:.3f}")

        with col_expand:
            if st.button(f"🔍 View All ({cluster_size})", key=f"expand_{cluster_id}", use_container_width=True):
                show_all_cluster_faces(cluster_id, cluster_size, faces_df, crops_dir)

        with col_identity:
            # Get current identity
            current_identity = labels_df.loc[labels_df['cluster_id'] == cluster_id, 'corrected_identity'].iloc[0]

            # Identity input with suggestions
            new_identity = st.selectbox(
                "Corrected Identity",
                options=[''] + st.session_state.identity_suggestions + ['[New Identity]'],
                index=st.session_state.identity_suggestions.index(current_identity) + 1 if current_identity in st.session_state.identity_suggestions else 0,
                key=f"select_{cluster_id}"
            )

            # Handle new identity
            if new_identity == '[New Identity]':
                new_identity = st.text_input(
                    "Enter new identity",
                    value=current_identity,
                    key=f"text_{cluster_id}"
                )
            elif new_identity == '':
                new_identity = current_identity

            # Update if changed
            if new_identity != current_identity:
                labels_df.loc[labels_df['cluster_id'] == cluster_id, 'corrected_identity'] = new_identity

                # Add to suggestions if new
                if new_identity not in st.session_state.identity_suggestions:
                    st.session_state.identity_suggestions.append(new_identity)
                    st.session_state.identity_suggestions.sort()

                changes_made = True

        # Display face crops (preview - first 20 faces)
        if crops_dir.exists():
            preview_count = 20
            images = get_cluster_face_images(cluster_id, faces_df, crops_dir, max_faces=preview_count)

            if len(images) > 0:
                # Show preview info
                if cluster_size > preview_count:
                    st.caption(f"Showing first {len(images)} of {cluster_size} faces. Click 'View All' button above to see all faces.")

                # Display in grid
                cols_per_row = 10
                n_rows = (len(images) + cols_per_row - 1) // cols_per_row

                for row in range(n_rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        img_idx = row * cols_per_row + col_idx
                        if img_idx < len(images):
                            face_id, img = images[img_idx]
                            with cols[col_idx]:
                                st.image(img, caption=f"{face_id}", use_container_width=True)
            else:
                st.info(f"No face crops found for cluster {cluster_id}")
        else:
            st.info("Face crops directory not available")

    # Save button
    st.markdown("---")
    col_save, col_reset, col_stats = st.columns([1, 1, 2])

    with col_save:
        if st.button("💾 Save Labels", type="primary", use_container_width=True):
            labels_path, metadata_path = save_labels(export_dir, labels_df)
            st.success(f"✅ Saved to:\n- {labels_path.name}\n- {metadata_path.name}")
            st.session_state.labels_df = labels_df  # Update session state
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset All", use_container_width=True):
            # Reset to default (cluster IDs)
            labels_df['corrected_identity'] = [f"person_{i}" for i in labels_df['cluster_id']]
            st.session_state.labels_df = labels_df
            st.rerun()

    with col_stats:
        if changes_made:
            st.info("⚠️ Unsaved changes")

    # Instructions
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        ### Labeling Workflow

        1. **Load Data**: Enter the export directory path and click "Load Data"
        2. **Review Clusters**: Look at the face images in each cluster
        3. **Assign Identities**:
           - Select existing identity from dropdown
           - Or choose "[New Identity]" and enter a new name
           - Use consistent names (e.g., "person_A", "person_B", "noise")
        4. **Save Progress**: Click "Save Labels" to save your work
        5. **Next Steps**: Use saved `corrected_labels.csv` for training

        ### Tips

        - **View All Faces**: Click the "🔍 View All" button to see all faces in a cluster (opens modal dialog)
        - **Noise clusters**: Label as "noise" if faces don't belong together
        - **Same person**: Give all clusters of the same person the same identity
        - **Pagination**: Use page controls to navigate through many clusters
        - **Sorting**: Sort by size to handle large clusters first

        ### After Labeling

        Run the training script:
        ```bash
        python scripts/train_merge_classifier.py \\
            --candidate-pairs {export_dir}/candidate_pairs.csv \\
            --corrected-labels {export_dir}/corrected_labels.csv \\
            --output {export_dir}/model
        ```
        """.format(export_dir=export_dir))

    # TAB 2: Merge Analysis
    if tab2 is not None:
        with tab2:
            show_merge_analysis(
                merge_decisions_df,
                faces_df,
                clusters_df,
                crops_dir,
                cluster_lineage,
                pre_merge_clusters_df
            )

    # TAB 3: Pre/Post Comparison
    if tab3 is not None:
        with tab3:
            show_pre_post_comparison(pre_merge_clusters_df, clusters_df, cluster_lineage, faces_df, crops_dir)

    # TAB 4: Pre-Cluster Analysis
    if tab4 is not None:
        with tab4:
            show_pre_cluster_analysis(
                pre_merge_clusters_df,
                faces_df,
                crops_dir,
                knn_graph_data
            )

else:
    st.info("👈 Enter export directory path in sidebar and click 'Load Data' to begin")

    st.markdown("""
    ### Getting Started

    1. First, export clustering data:
       ```bash
       python scripts/export_clustering_data.py \\
           --embeddings path/to/embeddings.npy \\
           --output results/face_clustering_training/dataset1
       ```

    2. Then load the output directory in this app

    3. Assign corrected identities to each cluster

    4. Save labels and proceed to training
    """)
