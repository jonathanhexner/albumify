"""
Test notebook export functionality.
"""
from pathlib import Path
from sim_bench.analysis.export import export_notebook_to_pdf

# Test with methods_comparison notebook
notebook_path = "sim_bench/analysis/methods_comparison.ipynb"

if Path(notebook_path).exists():
    print(f"Testing export of {notebook_path}")
    print("="*80)
    try:
        output_path = export_notebook_to_pdf(
            notebook_path,
            output_dir="sim_bench/analysis/reports",
            timestamp=True,
            prefix="test"
        )
        print("="*80)
        print(f"SUCCESS! Output: {output_path}")
    except Exception as e:
        print("="*80)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"ERROR: Notebook not found: {notebook_path}")
