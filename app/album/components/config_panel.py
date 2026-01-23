"""Configuration panel component."""

from typing import Dict, Any
import streamlit as st


def render_config_panel() -> Dict[str, Any]:
    """
    Render album configuration panel in sidebar.
    
    Returns:
        Configuration overrides dictionary
    """
    st.header("⚙️ Configuration")

    config = {'album': {}}

    # Quality thresholds
    with st.expander("📊 Quality Thresholds", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            min_iqa = st.slider("Min IQA", 0.0, 1.0, 0.3, 0.05)
            min_sharpness = st.slider("Min Sharpness", 0.0, 1.0, 0.2, 0.05)
        with col2:
            min_ava = st.slider("Min AVA", 1.0, 10.0, 4.0, 0.5)

        config['album']['quality'] = {
            'min_iqa_score': min_iqa,
            'min_ava_score': min_ava,
            'min_sharpness': min_sharpness,
        }

    # Portrait preferences
    with st.expander("👤 Portrait Preferences", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            require_eyes_open = st.checkbox("Require Eyes Open", True)
            prefer_smiling = st.checkbox("Prefer Smiling", True)
        with col2:
            smile_weight = st.slider("Smile Weight", 0.0, 1.0, 0.3, 0.1)
            eyes_weight = st.slider("Eyes Weight", 0.0, 1.0, 0.4, 0.1)

        config['album']['portrait'] = {
            'require_eyes_open': require_eyes_open,
            'prefer_smiling': prefer_smiling,
            'smile_importance': smile_weight,
            'eyes_open_importance': eyes_weight,
        }

    # Selection weights
    with st.expander("✨ Selection Weights", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            images_per_cluster = st.number_input("Images/Cluster", 1, 10, 1)
            ava_weight = st.slider("Aesthetic", 0.0, 1.0, 0.5, 0.1)
        with col2:
            iqa_weight = st.slider("Quality", 0.0, 1.0, 0.2, 0.1)
            portrait_weight = st.slider("Portrait", 0.0, 1.0, 0.3, 0.1)

        use_siamese = st.checkbox("Use Siamese Tiebreaker", True)

        total = ava_weight + iqa_weight + portrait_weight
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f}, should be 1.0")

        config['album']['selection'] = {
            'images_per_cluster': images_per_cluster,
            'ava_weight': ava_weight,
            'iqa_weight': iqa_weight,
            'portrait_weight': portrait_weight,
            'use_siamese_tiebreaker': use_siamese,
        }

    # Clustering
    with st.expander("🔍 Clustering", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cluster_method = st.selectbox("Method", ['hdbscan', 'dbscan', 'kmeans'])
            min_cluster = st.number_input("Min Size", 2, 20, 3)
        with col2:
            feature_method = st.selectbox("Features", ['dinov2', 'openclip', 'resnet50'])

        config['album']['clustering'] = {
            'method': cluster_method,
            'feature_method': feature_method,
            'min_cluster_size': min_cluster,
        }

    # Performance
    with st.expander("⚡ Performance", expanded=False):
        enable_preprocessing = st.checkbox("Thumbnail Preprocessing", True)
        num_workers = st.slider("Workers", 1, 8, 4) if enable_preprocessing else 4

        config['album']['preprocessing'] = {
            'enabled': enable_preprocessing,
            'num_workers': num_workers,
            'cache_thumbnails': True
        }

    # Export
    with st.expander("📤 Export", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Format", ['folder', 'zip'])
        with col2:
            organize_by_cluster = st.checkbox("By Cluster", True)
            include_thumbnails = st.checkbox("Thumbnails", True)

        config['album']['export'] = {
            'format': export_format,
            'organize_by_cluster': organize_by_cluster,
            'include_thumbnails': include_thumbnails,
        }

    return config
