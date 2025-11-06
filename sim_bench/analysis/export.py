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
        Tries WebPDFExporter first (requires Chrome/Chromium), then falls back to
        HTMLExporter if WebPDFExporter is not available.

        Install with: pip install nbconvert[webpdf]
        Or for LaTeX-based PDF: pip install nbconvert[pdf]

    Example:
        export_notebook_to_pdf("method_analysis.ipynb", prefix="deep")
        # Generates: deep_method_analysis_20251023_143022.pdf (or .html)
    """
    try:
        import nbconvert
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

    # Load and execute notebook to ensure all outputs (including plots) are captured
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    print(f"Loading notebook: {notebook_path.name}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Execute notebook to generate fresh outputs
    print(f"Executing notebook (this may take a few minutes)...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        print(f"Notebook executed successfully")
    except Exception as e:
        print(f"Warning: Notebook execution failed: {e}")
        print(f"Proceeding with existing outputs...")

    # Try WebPDFExporter first (best for plots), then fall back to HTML
    print(f"Converting to PDF...")

    try:
        from nbconvert import WebPDFExporter
        print(f"Using WebPDFExporter (Chrome/Chromium-based)...")
        exporter = WebPDFExporter()
        output_name = "_".join(parts) + ".pdf"
        output_path = output_dir / output_name
        (body, resources) = exporter.from_notebook_node(nb)

        with open(output_path, 'wb') as f:
            f.write(body)

        print(f"PDF saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"WebPDFExporter failed: {e}")
        print(f"Falling back to HTML export...")

        try:
            from nbconvert import HTMLExporter
            exporter = HTMLExporter()
            output_name = "_".join(parts) + ".html"
            output_path = output_dir / output_name
            (body, resources) = exporter.from_notebook_node(nb)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(body)

            print(f"HTML saved to: {output_path}")
            print(f"Note: HTML export used because WebPDFExporter requires Chrome/Chromium")
            print(f"      You can open the HTML file and print to PDF from your browser")
            return output_path

        except Exception as e2:
            print(f"HTMLExporter also failed: {e2}")
            raise RuntimeError(f"Could not export notebook: WebPDF failed ({e}), HTML failed ({e2})")


def quick_export(notebook_name: str = "method_analysis.ipynb", experiment_name: str = None) -> Path:
    """
    Quick export helper for analysis notebooks.

    Args:
        notebook_name: Name of the notebook in sim_bench/analysis/
        experiment_name: Optional experiment identifier to add to filename

    Returns:
        Path to exported PDF or HTML file
    """
    notebook_path = Path(__file__).parent / notebook_name

    if experiment_name:
        # Use export_notebook_to_pdf with experiment_name as prefix
        output_dir = notebook_path.parent / "reports"
        return export_notebook_to_pdf(
            str(notebook_path),
            output_dir=str(output_dir),
            timestamp=True,
            prefix=experiment_name
        )
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

