"""
Generate a comprehensive CSV summary of all bags in the comparison data directory.

Usage:
    python scripts/generate_bag_summary.py
    python scripts/generate_bag_summary.py --output custom_summary.csv
    python scripts/generate_bag_summary.py --comparison-dir /path/to/comparison_data
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.comparison_analysis import generate_multi_bag_summary_csv


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive CSV summary of all analyzed bags'
    )
    
    parser.add_argument(
        '--comparison-dir',
        type=str,
        default='/Volumes/LaCie/SOLAQUA/comparison_data',
        help='Directory containing *_raw_comparison.csv files (default: /Volumes/LaCie/SOLAQUA/comparison_data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: <comparison-dir>/bag_summary.csv)'
    )
    
    args = parser.parse_args()
    
    comparison_dir = Path(args.comparison_dir)
    
    if not comparison_dir.exists():
        print(f"Error: Comparison directory not found: {comparison_dir}")
        sys.exit(1)
    
    print(f"Comparison directory: {comparison_dir}")
    print(f"Scanning for *_raw_comparison.csv files...\n")
    
    # Generate summary
    output_path = generate_multi_bag_summary_csv(
        comparison_dir,
        output_path=args.output
    )
    
    if output_path:
        print(f"\n✓ Success! Summary saved to: {output_path}")
        print(f"\nYou can now:")
        print(f"  - Open in Excel/Numbers for analysis")
        print(f"  - Load in pandas: pd.read_csv('{output_path}')")
        print(f"  - Filter/sort to compare bags")
    else:
        print("\n✗ Failed to generate summary")
        sys.exit(1)


if __name__ == '__main__':
    main()
