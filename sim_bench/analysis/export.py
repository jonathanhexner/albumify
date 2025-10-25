"""Export utilities for analysis notebooks."""
from pathlib import Path
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt
import shutil


def export_notebook_to_pdf(
    notebook_path: str,
    output_dir: Optional[str] = None,
    timestamp: bool = True,
    prefix: Optional[str] = None
) -> Path:
    """
    Export a Jupyter notebook to PDF.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_dir: Output directory (defaults to same dir as notebook)
        timestamp: If True, add timestamp to filename
        prefix: Optional prefix for filename (e.g., method name like "deep", "emd")
    
    Returns:
        Path to the generated PDF file
    
    Note:
        Requires: pip install nbconvert[webpdf] or nbconvert[pdf]
        For webpdf: Uses chromium/chrome for conversion
        For pdf: Requires LaTeX installation
    
    Example:
        export_notebook_to_pdf("method_analysis.ipynb", prefix="deep")
        # Generates: deep_method_analysis_20251023_143022.pdf
    """
    try:
        import nbconvert
        from nbconvert import PDFExporter
    except ImportError:
        raise ImportError(
            "nbconvert not installed. Install with: pip install nbconvert[webpdf]"
        )
    
    notebook_path = Path(notebook_path)
    output_dir = Path(output_dir) if output_dir else notebook_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with optional prefix
    base_name = notebook_path.stem
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(base_name)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(ts)
    
    output_name = "_".join(parts) + ".pdf"
    output_path = output_dir / output_name
    
    # Export to PDF
    pdf_exporter = PDFExporter()
    (body, resources) = pdf_exporter.from_filename(str(notebook_path))
    
    with open(output_path, 'wb') as f:
        f.write(body)
    
    return output_path


def quick_export(notebook_name: str = "method_analysis.ipynb", experiment_name: str = None) -> Path:
    """
    Quick export helper for analysis notebooks.
    
    Args:
        notebook_name: Name of the notebook in sim_bench/analysis/
        experiment_name: Optional experiment identifier to add to filename
    
    Returns:
        Path to exported PDF
    """
    notebook_path = Path(__file__).parent / notebook_name
    
    if experiment_name:
        base = notebook_path.stem
        output_name = f"{base}_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = notebook_path.parent / "reports" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import nbconvert
        pdf_exporter = nbconvert.PDFExporter()
        (body, resources) = pdf_exporter.from_filename(str(notebook_path))
        
        with open(output_path, 'wb') as f:
            f.write(body)
        
        return output_path
    else:
        return export_notebook_to_pdf(str(notebook_path))


def archive_notebook(
    notebook_path: str,
    output_dir: Optional[str] = None,
    timestamp: bool = True,
    prefix: Optional[str] = None
) -> Path:
    """
    Archive (copy) a notebook to the output directory for future reference.
    
    This preserves the exact configuration and analysis code used for an experiment,
    making results more reproducible.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_dir: Output directory (defaults to same dir as notebook)
        timestamp: If True, add timestamp to filename
        prefix: Optional prefix for filename (e.g., method name)
    
    Returns:
        Path to the archived notebook file
    
    Example:
        archive_notebook(
            "sim_bench/analysis/method_analysis.ipynb",
            output_dir="outputs/.../analysis_reports",
            prefix="deep"
        )
        # Creates: deep_method_analysis_20251023_143022.ipynb
    """
    notebook_path = Path(notebook_path)
    output_dir = Path(output_dir) if output_dir else notebook_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with optional prefix and timestamp
    base_name = notebook_path.stem
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(base_name)
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(ts)
    
    output_name = "_".join(parts) + ".ipynb"
    output_path = output_dir / output_name
    
    # Copy notebook
    shutil.copy2(notebook_path, output_path)
    
    return output_path


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    notebook_type: str = "feature_exploration",
    method_name: Optional[str] = None,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> Path:
    """
    Save a matplotlib figure to an organized directory structure.
    
    Args:
        fig: Matplotlib figure to save
        output_dir: Base output directory (typically experiment_dir / "analysis_reports")
        filename: Base filename (e.g., "discriminability_boxplot")
        notebook_type: Type of notebook/analysis (e.g., "feature_exploration", "method_analysis")
        method_name: Optional method name prefix (e.g., "deep", "sift_bovw", "emd")
        dpi: Output resolution (default: 300)
        bbox_inches: Bounding box setting (default: 'tight')
    
    Returns:
        Path to saved figure
    
    Directory Structure:
        output_dir/
          plots/
            {notebook_type}/
              {method_name}_{filename}.png  (if method_name provided)
              {filename}.png               (if method_name is None)
    
    Example:
        save_figure(
            fig=plt.gcf(),
            output_dir=Path("outputs/baseline_runs/.../analysis_reports"),
            filename="discriminability_boxplot",
            notebook_type="feature_exploration",
            method_name="deep"
        )
        # Saves to: .../analysis_reports/plots/feature_exploration/deep_discriminability_boxplot.png
    """
    plots_dir = output_dir / "plots" / notebook_type
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename with optional method prefix
    parts = []
    if method_name:
        parts.append(method_name)
    parts.append(filename)
    
    full_filename = "_".join(parts) + ".png"
    output_path = plots_dir / full_filename
    
    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    
    return output_path

