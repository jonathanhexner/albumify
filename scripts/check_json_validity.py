"""
Check JSON file validity and provide diagnostics.

Usage:
    python scripts/check_json_validity.py results/face_clustering_benchmark/benchmark_*.json
"""

import sys
import json
from pathlib import Path


def check_json_file(filepath: Path):
    """Check if JSON file is valid and provide diagnostics."""
    print(f"\nChecking: {filepath}")
    print("=" * 70)
    
    if not filepath.exists():
        print(f"❌ File does not exist!")
        return False
    
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✓ File size: {len(content):,} characters ({len(content)/1024:.1f} KB)")
        
        # Try to parse
        data = json.loads(content)
        print(f"✓ JSON is valid!")
        
        # Show structure
        if isinstance(data, dict):
            print(f"✓ Top-level keys: {list(data.keys())}")
            if 'total_faces' in data:
                print(f"✓ Total faces: {data['total_faces']}")
            if 'methods' in data:
                print(f"✓ Methods: {list(data['methods'].keys())}")
                for method, results in data['methods'].items():
                    if 'stats' in results:
                        stats = results['stats']
                        print(f"  - {method}: {stats.get('n_clusters', '?')} clusters")
        
        print(f"\n✅ File is valid and complete!\n")
        return True
        
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON Parse Error!")
        print(f"   Line: {e.lineno}, Column: {e.colno}")
        print(f"   Character position: {e.pos}")
        print(f"   Error: {e.msg}")
        
        # Show context around error
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            start = max(0, e.pos - 200)
            end = min(len(content), e.pos + 200)
            context = content[start:end]
            
            # Find relative position in context
            rel_pos = e.pos - start
            
            print(f"\nContext around error (±200 chars):")
            print("=" * 70)
            print(context[:rel_pos])
            print(">>> ERROR HERE <<<")
            print(context[rel_pos:])
            print("=" * 70)
            
            # Check if file is truncated
            if e.pos == len(content):
                print("\n⚠️  File appears to be TRUNCATED (error at end of file)")
                print("   The JSON file was likely interrupted while being written.")
                print("   Solution: Re-run the benchmark script.")
            
        except Exception as show_err:
            print(f"   (Could not show context: {show_err})")
        
        print(f"\n❌ File is invalid or incomplete!\n")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}\n")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_json_validity.py <json_file>")
        print("\nExample:")
        print("  python scripts/check_json_validity.py results/face_clustering_benchmark/benchmark_*.json")
        return 1
    
    # Handle glob patterns
    import glob
    files = []
    for pattern in sys.argv[1:]:
        files.extend(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching: {sys.argv[1:]}")
        return 1
    
    all_valid = True
    for filepath in files:
        if not check_json_file(Path(filepath)):
            all_valid = False
    
    if all_valid:
        print(f"✅ All {len(files)} file(s) are valid!")
        return 0
    else:
        print(f"❌ Some files are invalid. See details above.")
        return 1


if __name__ == '__main__':
    exit(main())
