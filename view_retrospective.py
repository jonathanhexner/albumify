"""Simple script to view the project retrospective in a browser."""

import webbrowser
import sys
from pathlib import Path


def main():
    """Open the retrospective HTML in default browser."""
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    html_file = Path(__file__).parent / "RETROSPECTIVE.html"
    
    if not html_file.exists():
        print("Error: RETROSPECTIVE.html not found")
        print("Please ensure RETROSPECTIVE.html exists in the project root")
        return
    
    print("Opening sim-bench Retrospective...")
    print(f"File: {html_file.absolute()}")
    
    webbrowser.open(html_file.as_uri())
    
    print("Opened in your default browser")
    print("\nAlternatively:")
    print(f"  - View HTML: file://{html_file.absolute()}")
    print(f"  - View Markdown: RETROSPECTIVE.md")
    print(f"  - In Streamlit: streamlit run app/ui/retrospective_page.py")


if __name__ == "__main__":
    main()
