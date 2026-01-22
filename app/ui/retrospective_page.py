"""Retrospective page for Streamlit app showing project timeline and achievements."""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path


def render_retrospective_page() -> None:
    """Render the project retrospective page."""
    st.set_page_config(
        page_title="sim-bench Retrospective",
        page_icon="📊",
        layout="wide"
    )
    
    html_file = Path(__file__).parent.parent.parent / "RETROSPECTIVE.html"
    
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        components.html(html_content, height=4000, scrolling=True)
    else:
        st.error(f"Retrospective HTML file not found at {html_file}")
        st.info("Run from the project root or ensure RETROSPECTIVE.html exists")


def render_retrospective_tab() -> None:
    """Render retrospective as a tab within existing app."""
    html_file = Path(__file__).parent.parent.parent / "RETROSPECTIVE.html"
    
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        components.html(html_content, height=3000, scrolling=True)
    else:
        st.error("⚠️ Retrospective HTML not found")
        st.markdown("""
        Please ensure `RETROSPECTIVE.html` exists in the project root.
        
        You can also view the markdown version: [RETROSPECTIVE.md](RETROSPECTIVE.md)
        """)


def render_retrospective_summary() -> None:
    """Render a compact summary card of key achievements."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
    ">
        <h2 style="margin: 0 0 20px 0;">📊 Project Highlights</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2em; font-weight: bold;">6+</div>
                <div style="font-size: 0.9em;">Months</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2em; font-weight: bold;">15+</div>
                <div style="font-size: 0.9em;">Methods</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2em; font-weight: bold;">5</div>
                <div style="font-size: 0.9em;">Datasets</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2em; font-weight: bold;">89.9%</div>
                <div style="font-size: 0.9em;">Best Accuracy</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        ### Recent Achievements
        
        - ✅ **Unified Quality Benchmark** - Compared Siamese, AVA, and IQA models
        - ✅ **Learned CLIP Prompts** - Extracted from 34,827 user feedback reasons
        - ✅ **App Refactoring** - Production-grade architecture with full type safety
        - ✅ **Repository Cleanup** - 97% reduction in root clutter
        """)
    
    with col2:
        if st.button("📊 View Full Timeline", use_container_width=True):
            st.session_state.show_full_retrospective = True
        
        if st.button("📚 Documentation", use_container_width=True):
            st.markdown("[View README](README.md)")


if __name__ == "__main__":
    render_retrospective_page()
