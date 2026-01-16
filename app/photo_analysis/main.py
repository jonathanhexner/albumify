"""
Streamlit application for photo analysis and organization.
Matches the capabilities of photo_analysis_demo.py script.
"""

import streamlit as st
import sys
import logging
from pathlib import Path
import tempfile
import shutil
from typing import List, Optional, Dict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.image_processing import ThumbnailGenerator
from sim_bench.photo_analysis import PhotoAnalysisPipeline, generate_html_report, CLIPTagger
from sim_bench.config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Photo Analysis",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Photo Analysis & Organization")
st.markdown("Analyze photos with CLIP tagging, face detection, and landmark recognition")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Image input
    st.subheader("Input")
    uploaded_files = st.file_uploader(
        "Upload photos",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    # Or directory input
    input_dir = st.text_input("Or specify directory path", value="")
    
    # Processing options
    st.subheader("Processing Options")
    max_images = st.number_input("Max images to process", min_value=1, max_value=1000, value=20)
    thumbnail_sizes = st.multiselect(
        "Thumbnail sizes",
        options=['tiny', 'small', 'medium', 'large'],
        default=['tiny']
    )
    batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=8)
    
    # Analysis options
    st.subheader("Analysis Options")
    use_specialized_models = st.checkbox("Use Specialized Models", value=True)
    enable_faces = st.checkbox("Enable Face Detection", value=True)
    enable_landmarks = st.checkbox("Enable Landmark Detection", value=True)
    
    # Device selection
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    
    # Generate report
    generate_report = st.checkbox("Generate HTML Report", value=True)


def _get_sample_images(samples_dir: Path, max_images: int) -> Optional[List[Path]]:
    """Get sample images from directory (matching demo script)."""
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        logger.warning(f"Directory not found: {samples_dir}")
        return None
    
    images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.jpeg")) + list(samples_dir.glob("*.png"))
    return images[:max_images] if images else None


# Main content
if uploaded_files or input_dir:
    if st.button("Analyze Photos", type="primary"):
        with st.spinner("Processing photos..."):
            # Collect image paths (matching demo script approach)
            image_paths = []
            temp_dir = None
            
            if uploaded_files:
                # Save uploaded files to temp directory
                temp_dir = Path(tempfile.mkdtemp())
                for uploaded_file in uploaded_files[:max_images]:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    image_paths.append(file_path)
            elif input_dir:
                input_path = Path(input_dir)
                if input_path.exists():
                    sample_images = _get_sample_images(input_path, max_images)
                    if sample_images:
                        image_paths = sample_images
                    else:
                        st.error(f"No images found in: {input_dir}")
                        st.stop()
                else:
                    st.error(f"Directory not found: {input_dir}")
                    st.stop()
            
            if not image_paths:
                st.warning("No images found")
                st.stop()
            
            st.info(f"Processing {len(image_paths)} images")
            
            # Step 1: Generate thumbnails (matching demo script)
            st.subheader("Step 1/3: Generating Thumbnails")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            generator = ThumbnailGenerator(cache_dir=".cache/app_thumbnails")
            thumbnail_results = generator.process_batch(
                [str(p) for p in image_paths],
                sizes=thumbnail_sizes if thumbnail_sizes else ['tiny'],
                num_workers=4,
                verbose=False
            )
            
            # Get tiny thumbnails for analysis (matching demo script)
            tiny_paths = []
            original_mapping = {}
            
            for img_path in image_paths:
                img_str = str(img_path)
                if img_str in thumbnail_results:
                    tiny_thumb = thumbnail_results[img_str].get('tiny')
                    if tiny_thumb:
                        tiny_paths.append(tiny_thumb)
                        original_mapping[str(tiny_thumb)] = img_path
            
            if not tiny_paths:
                st.error("Failed to generate thumbnails")
                st.stop()
            
            progress_bar.progress(33)
            status_text.text(f"Generated {len(tiny_paths)} thumbnails")
            
            # Step 2: Analyze with CLIP (matching demo script)
            st.subheader("Step 2/3: Analyzing with CLIP")
            
            if use_specialized_models:
                # Use pipeline with specialized models
                face_config = {'backend': 'deepface', 'device': device} if enable_faces else None
                landmark_config = {'device': device} if enable_landmarks else None
                
                pipeline = PhotoAnalysisPipeline(
                    clip_config={'device': device},
                    face_config=face_config,
                    landmark_config=landmark_config
                )
                
                combined_results = pipeline.analyze_with_specialized([str(p) for p in tiny_paths])
                
                # Extract CLIP results for report (matching demo format)
                clip_results = {
                    str(path): results['clip_analysis']
                    for path, results in combined_results.items()
                }
            else:
                # Use CLIPTagger directly (matching demo script exactly)
                tagger = CLIPTagger(device=device)
                clip_results = tagger.analyze_batch(
                    [str(p) for p in tiny_paths],
                    batch_size=batch_size,
                    verbose=False
                )
            
            progress_bar.progress(66)
            status_text.text(f"Analyzed {len(clip_results)} images")
            
            # Step 3: Generate HTML report (matching demo script)
            st.subheader("Step 3/3: Generating HTML Report")
            
            if generate_report:
                output_file = Path("outputs/app_analysis_report.html")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                report_path = generate_html_report(
                    analysis_results=clip_results,
                    original_images=original_mapping,
                    output_path=output_file,
                    title="Photo Analysis Report"
                )
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
                st.success(f"✅ Report generated: {report_path}")
                
                # Display statistics
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                total_images = len(clip_results)
                avg_importance = sum(
                    r.get('importance_score', 0) for r in clip_results.values()
                ) / total_images if total_images > 0 else 0
                
                with_faces = sum(
                    1 for r in clip_results.values()
                    if r.get('routing', {}).get('needs_face_detection', False)
                )
                with_landmarks = sum(
                    1 for r in clip_results.values()
                    if r.get('routing', {}).get('needs_landmark_detection', False)
                )
                
                col1.metric("Total Images", total_images)
                col2.metric("With Faces", with_faces)
                col3.metric("With Landmarks", with_landmarks)
                col4.metric("Avg Importance", f"{avg_importance:.3f}")
                
                # Display report in iframe
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                st.subheader("HTML Report")
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Download button
                with open(report_path, 'rb') as f:
                    st.download_button(
                        label="Download HTML Report",
                        data=f.read(),
                        file_name=report_path.name,
                        mime="text/html"
                    )
            else:
                progress_bar.progress(100)
                status_text.text("Analysis complete (report generation skipped)")
            
            # Cleanup temp files
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
else:
    st.info("👈 Upload photos or specify a directory path to get started")
    
    st.markdown("""
    ### Features (matching photo_analysis_demo.py)
    - **Thumbnail Generation**: Multi-resolution thumbnails with EXIF orientation handling
    - **CLIP Tagging**: 57 zero-shot prompts for comprehensive photo understanding
    - **Routing**: Automatic decision on which specialized models to use
    - **Face Detection**: Automatic face detection and embedding extraction (optional)
    - **Landmark Recognition**: Place/landmark identification (optional)
    - **HTML Reports**: Visual analysis reports with categorized tags and routing information
    
    ### Workflow
    1. Generate thumbnails (tiny size for analysis)
    2. Analyze with CLIP (batch processing)
    3. Generate HTML report with original images and analysis results
    """)

