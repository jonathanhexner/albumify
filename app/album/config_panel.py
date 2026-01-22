"""
Configuration panel UI for album organization.
"""

from typing import Dict, Any


def render_album_config() -> Dict[str, Any]:
    """
    Render album configuration panel.
    
    Returns:
        Configuration overrides dictionary
    """
    import streamlit as st
    
    st.header("⚙️ Album Configuration")
    
    config_overrides = {'album': {}}
    
    with st.expander("📊 Quality Thresholds", expanded=True):
        st.markdown("Set minimum quality thresholds for filtering images.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_iqa = st.slider(
                "Min Technical Quality (IQA)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum technical quality score (0-1)"
            )
            
            min_sharpness = st.slider(
                "Min Sharpness",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Minimum sharpness score (0-1)"
            )
        
        with col2:
            min_ava = st.slider(
                "Min Aesthetic Score (AVA)",
                min_value=1.0,
                max_value=10.0,
                value=4.0,
                step=0.5,
                help="Minimum aesthetic score (1-10)"
            )
        
        config_overrides['album']['quality'] = {
            'min_iqa_score': min_iqa,
            'min_ava_score': min_ava,
            'min_sharpness': min_sharpness,
        }
    
    with st.expander("👤 Portrait Preferences", expanded=True):
        st.markdown("Configure portrait-specific filtering and preferences.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            require_eyes_open = st.checkbox(
                "Require Eyes Open",
                value=True,
                help="Filter out portraits with closed eyes"
            )
            
            prefer_smiling = st.checkbox(
                "Prefer Smiling",
                value=True,
                help="Give bonus to smiling portraits in selection"
            )
        
        with col2:
            smile_weight = st.slider(
                "Smile Importance",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Weight for smile in portrait scoring"
            )
            
            eyes_weight = st.slider(
                "Eyes Open Importance",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.1,
                help="Weight for eyes open in portrait scoring"
            )
        
        config_overrides['album']['portrait'] = {
            'require_eyes_open': require_eyes_open,
            'prefer_smiling': prefer_smiling,
            'smile_importance': smile_weight,
            'eyes_open_importance': eyes_weight,
        }
    
    with st.expander("🔍 Clustering Settings", expanded=False):
        st.markdown("Configure image clustering parameters.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_method = st.selectbox(
                "Clustering Method",
                options=['hdbscan', 'dbscan', 'kmeans'],
                index=0,
                help="Algorithm for grouping similar images"
            )
            
            min_cluster_size = st.number_input(
                "Min Cluster Size",
                min_value=2,
                max_value=20,
                value=3,
                step=1,
                help="Minimum images per cluster"
            )
        
        with col2:
            feature_method = st.selectbox(
                "Feature Method",
                options=['dinov2', 'openclip', 'resnet50'],
                index=0,
                help="Feature extraction method"
            )
        
        config_overrides['album']['clustering'] = {
            'method': cluster_method,
            'feature_method': feature_method,
            'min_cluster_size': min_cluster_size,
        }
    
    with st.expander("✨ Selection Settings", expanded=False):
        st.markdown("Configure best image selection criteria.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            images_per_cluster = st.number_input(
                "Images Per Cluster",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Number of best images to select per cluster"
            )
            
            ava_weight = st.slider(
                "Aesthetic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Weight for aesthetic score in selection"
            )
        
        with col2:
            iqa_weight = st.slider(
                "Quality Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Weight for technical quality in selection"
            )
            
            portrait_weight = st.slider(
                "Portrait Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Weight for portrait metrics in selection"
            )
        
        use_siamese = st.checkbox(
            "Use Siamese Tiebreaker",
            value=True,
            help="Use Siamese model for close decisions"
        )
        
        total_weight = ava_weight + iqa_weight + portrait_weight
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}, should be 1.0")
        
        config_overrides['album']['selection'] = {
            'images_per_cluster': images_per_cluster,
            'ava_weight': ava_weight,
            'iqa_weight': iqa_weight,
            'portrait_weight': portrait_weight,
            'use_siamese_tiebreaker': use_siamese,
        }
    
    with st.expander("⚡ Performance Settings", expanded=False):
        st.markdown("Configure preprocessing and performance options.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_preprocessing = st.checkbox(
                "Enable Thumbnail Preprocessing",
                value=True,
                help="Pre-generate thumbnails for 50%+ speedup"
            )
        
        with col2:
            if enable_preprocessing:
                num_workers = st.slider(
                    "Parallel Workers",
                    min_value=1,
                    max_value=8,
                    value=4,
                    help="Number of parallel workers for thumbnail generation"
                )
            else:
                num_workers = 4
        
        config_overrides['album']['preprocessing'] = {
            'enabled': enable_preprocessing,
            'num_workers': num_workers,
            'cache_thumbnails': True
        }
    
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.markdown("Advanced clustering and filtering options.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=1,
                max_value=10,
                value=3,
                help="Minimum images per cluster. 1 = allow singletons, higher = fewer/larger clusters"
            )
        
        with col2:
            st.info("ℹ️ Singletons (size=1) are kept as individual clusters")
        
        config_overrides['album']['clustering']['min_cluster_size'] = min_cluster_size
    
    with st.expander("📤 Export Settings", expanded=False):
        st.markdown("Configure export options.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                options=['folder', 'zip'],
                index=0,
                help="Output format for selected images"
            )
        
        with col2:
            organize_by_cluster = st.checkbox(
                "Organize by Cluster",
                value=True,
                help="Create separate folders per cluster"
            )
            
            include_thumbnails = st.checkbox(
                "Include Thumbnails",
                value=True,
                help="Generate thumbnails in export"
            )
        
        config_overrides['album']['export'] = {
            'format': export_format,
            'organize_by_cluster': organize_by_cluster,
            'include_thumbnails': include_thumbnails,
        }
    
    return config_overrides
