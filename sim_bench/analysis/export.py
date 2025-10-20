"""Export utilities for analysis notebooks."""
from pathlib import Path
from datetime import datetime
from typing import Optional


def export_notebook_to_pdf(
    notebook_path: str,
    output_dir: Optional[str] = None,
    timestamp: bool = True
) -> Path:
    """
    Export a Jupyter notebook to PDF.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_dir: Output directory (defaults to same dir as notebook)
        timestamp: If True, add timestamp to filename
    
    Returns:
        Path to the generated PDF file
    
    Note:
        Requires: pip install nbconvert[webpdf] or nbconvert[pdf]
        For webpdf: Uses chromium/chrome for conversion
        For pdf: Requires LaTeX installation
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
    
    # Generate output filename
    base_name = notebook_path.stem
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{base_name}_{ts}.pdf"
    else:
        output_name = f"{base_name}.pdf"
    
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

