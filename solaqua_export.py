#!/usr/bin/env python3
"""
SOLAQUA Complete Export Tool

This single script handles all SOLAQUA data export operations:
- CSV/Parquet export from data bags
- Video export (MP4) from video bags  
- Frame extraction (PNG sequences) from video bags
- Camera info export (YAML)
- Cone NPZ creation from sonar data

Usage:
    python3 solaqua_export.py --help
    python3 solaqua_export.py --data-dir data --exports-dir exports
    python3 solaqua_export.py --data-dir data --exports-dir exports --all
    python3 solaqua_export.py --data-dir data --exports-dir exports --csv-only
    python3 solaqua_export.py --data-dir data --exports-dir exports --video-only
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Import our consolidated utility module
from utils.dataset_export_utils import (
    save_all_topics_from_data_bags,
    list_topics_in_bag,
    export_all_video_bags_to_mp4,
    export_all_video_bags_to_png,
    export_camera_info_for_bag,
    list_camera_topics_in_bag,
    find_data_bags,
    find_video_bags,
)
import utils.sonar_utils as sonar_utils


# ------------------------------ Convenience listings ------------------------------

def load_data_index(out_dir: Path | str = "exports") -> pd.DataFrame:
    """
    Load the index created by save_all_topics_from_data_bags().
    Looks for exports/index_data_topics.csv (or .parquet).
    """
    out_dir = Path(out_dir)
    csv = out_dir / "index_data_topics.csv"
    pq  = out_dir / "index_data_topics.parquet"
    if csv.exists():
        return pd.read_csv(csv)
    if pq.exists():
        return pd.read_parquet(pq)
    raise FileNotFoundError(
        f"Could not find index_data_topics.csv or .parquet under {out_dir}. "
        "Run save_all_topics_from_data_bags() first."
    )


def list_exported_bag_stems(out_dir: Path | str = "exports",
                            bag_suffix: str = "_data") -> List[str]:
    """Return bag stems present in the export index, optionally filtered by suffix."""
    idx = load_data_index(out_dir)
    if bag_suffix:
        idx = idx[idx["bag"].str.endswith(bag_suffix)]
    return sorted(idx["bag"].unique().tolist())

class SOLAQUACompleteExporter:
    """
    Complete SOLAQUA data export tool that handles all export operations.
    """
    
    def __init__(self, data_dir: Union[str, Path], exports_dir: Union[str, Path] = "exports"):
        """
        Initialize the complete exporter.
        
        Args:
            data_dir: Directory containing *.bag files
            exports_dir: Directory where exports will be saved
        """
        self.data_dir = Path(data_dir)
        self.exports_dir = Path(exports_dir)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Set max string digits for large numbers
        sys.set_int_max_str_digits(100000)
        
        print(f"🌊 SOLAQUA Complete Export Tool")
        print(f"Data directory: {self.data_dir.resolve()}")
        print(f"Exports directory: {self.exports_dir.resolve()}")
        
        # Track results
        self.results = {
            'csv_export': {'success': False, 'files': 0, 'errors': []},
            'video_export': {'success': False, 'files': 0, 'errors': []},
            'frame_export': {'success': False, 'files': 0, 'errors': []},
            'camera_info': {'success': False, 'files': 0, 'errors': []},
            'npz_creation': {'success': False, 'files': 0, 'errors': []}
        }
    
    def discover_bags(self) -> Dict[str, List[Path]]:
        """Discover all available bag files."""
        data_bags = find_data_bags(self.data_dir, recursive=True)
        video_bags = find_video_bags(self.data_dir, recursive=True)
        
        print(f"\n📁 Discovered bag files:")
        print(f"   Data bags (*_data.bag): {len(data_bags)}")
        print(f"   Video bags (*_video.bag): {len(video_bags)}")
        
        if data_bags:
            print(f"   Sample data bags:")
            for bag in data_bags[:3]:
                print(f"     - {bag.name}")
            if len(data_bags) > 3:
                print(f"     ... and {len(data_bags) - 3} more")
        
        if video_bags:
            print(f"   Sample video bags:")
            for bag in video_bags[:3]:
                print(f"     - {bag.name}")
            if len(video_bags) > 3:
                print(f"     ... and {len(video_bags) - 3} more")
        
        return {'data_bags': data_bags, 'video_bags': video_bags}
    
    def inspect_bags(self, bags: Dict[str, List[Path]]) -> None:
        """Inspect topics in sample bags."""
        print(f"\n🔍 Inspecting bag contents:")
        
        # Inspect first data bag
        if bags['data_bags']:
            sample_data_bag = bags['data_bags'][0]
            print(f"\n   Data bag: {sample_data_bag.name}")
            try:
                topics = list_topics_in_bag(sample_data_bag)
                print(f"     Topics: {len(topics)}")
                for topic, msgtype in topics[:5]:
                    print(f"       - {topic} ({msgtype})")
                if len(topics) > 5:
                    print(f"       ... and {len(topics) - 5} more")
            except Exception as e:
                print(f"     ❌ Error inspecting: {e}")
        
        # Inspect first video bag
        if bags['video_bags']:
            sample_video_bag = bags['video_bags'][0]
            print(f"\n   Video bag: {sample_video_bag.name}")
            try:
                topics = list_camera_topics_in_bag(sample_video_bag)
                print(f"     Camera topics: {len(topics)}")
                for topic, msgtype in topics[:3]:
                    print(f"       - {topic} ({msgtype})")
                if len(topics) > 3:
                    print(f"       ... and {len(topics) - 3} more")
            except Exception as e:
                print(f"     ❌ Error inspecting: {e}")
    
    def export_csv_data(
        self, 
        file_format: str = "csv",
        include_video_sonar: bool = True
    ) -> bool:
        """
        Export CSV/Parquet data from data bags.
        
        Args:
            file_format: 'csv' or 'parquet'
            include_video_sonar: Whether to include sonar topics from video bags
            
        Returns:
            True if successful
        """
        print(f"\n📊 Exporting data to {file_format.upper()}...")
        print(f"   Include video sonar: {include_video_sonar}")
        
        try:
            export_index_df = save_all_topics_from_data_bags(
                self.data_dir,
                out_dir=self.exports_dir,
                file_format=file_format,
                recursive=True,
                include_video_sonar=include_video_sonar
            )
            
            self.results['csv_export']['success'] = True
            self.results['csv_export']['files'] = len(export_index_df)
            
            print(f"✅ CSV export completed successfully!")
            print(f"   Exported {len(export_index_df)} topic files")
            
            return True
            
        except Exception as e:
            error_msg = f"Error during CSV export: {e}"
            self.results['csv_export']['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def export_videos(
        self,
        target_fps: Optional[float] = None,
        limit_per_bag: Optional[int] = None
    ) -> bool:
        """
        Export videos (MP4) from video bags.
        
        Args:
            target_fps: Target FPS for videos (None for auto-detect)
            limit_per_bag: Limit number of videos per bag (not supported for MP4 export)
            
        Returns:
            True if successful
        """
        print(f"\n🎥 Exporting videos to MP4...")
        print(f"   Target FPS: {target_fps or 'auto-detect'}")
        if limit_per_bag:
            print(f"   Note: Video limit not supported for MP4 export, processing all videos")
        
        try:
            idx_mp4 = export_all_video_bags_to_mp4(
                self.data_dir,
                out_dir=self.exports_dir,
                recursive=True,
                target_fps=target_fps
            )
            
            self.results['video_export']['success'] = True
            self.results['video_export']['files'] = len(idx_mp4)
            
            print(f"✅ Video export completed successfully!")
            print(f"   Exported {len(idx_mp4)} MP4 files")
            
            return True
            
        except Exception as e:
            error_msg = f"Error during video export: {e}"
            self.results['video_export']['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def export_frames(
        self,
        stride: int = 10,
        limit_per_bag: Optional[int] = 100,
        resize_to: Optional[tuple] = None
    ) -> bool:
        """
        Export frame sequences (PNG) from video bags.
        
        Args:
            stride: Take every Nth frame
            limit_per_bag: Maximum frames per bag
            resize_to: Resize frames to (width, height) or None
            
        Returns:
            True if successful
        """
        print(f"\n🖼️  Exporting frame sequences to PNG...")
        print(f"   Frame stride: {stride} (every {stride}th frame)")
        print(f"   Limit per bag: {limit_per_bag or 'no limit'}")
        print(f"   Resize to: {resize_to or 'original size'}")
        
        try:
            idx_png = export_all_video_bags_to_png(
                self.data_dir,
                out_dir=self.exports_dir,
                recursive=True,
                stride=stride,
                limit=limit_per_bag,
                resize_to=resize_to
            )
            
            self.results['frame_export']['success'] = True
            self.results['frame_export']['files'] = len(idx_png)
            
            print(f"✅ Frame export completed successfully!")
            print(f"   Exported {len(idx_png)} frame sequences")
            
            return True
            
        except Exception as e:
            error_msg = f"Error during frame export: {e}"
            self.results['frame_export']['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def export_camera_info(self) -> bool:
        """
        Export camera info (YAML) from video bags.
        
        Returns:
            True if successful
        """
        print(f"\n📷 Exporting camera info to YAML...")
        
        try:
            video_bags = find_video_bags(self.data_dir, recursive=True)
            if not video_bags:
                print("⚠️  No video bags found for camera info export")
                return True  # Not an error, just no bags
            
            camera_info_dir = self.exports_dir / "camera_info"
            total_files = 0
            
            for bag in video_bags:
                try:
                    df_info = export_camera_info_for_bag(bag, out_dir=camera_info_dir)
                    total_files += len(df_info)
                    print(f"   Exported camera info from {bag.name}: {len(df_info)} files")
                except Exception as e:
                    print(f"   ⚠️  Error with {bag.name}: {e}")
            
            self.results['camera_info']['success'] = True
            self.results['camera_info']['files'] = total_files
            
            print(f"✅ Camera info export completed!")
            print(f"   Total YAML files: {total_files}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error during camera info export: {e}"
            self.results['camera_info']['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def create_cone_npz(
        self,
        max_bags: Optional[int] = None,
        cone_params: Optional[Dict] = None
    ) -> bool:
        """
        Create cone NPZ files from sonar data.
        
        Args:
            max_bags: Maximum number of bags to process
            cone_params: Parameters for cone generation
            
        Returns:
            True if successful
        """
        print(f"\n🔧 Creating cone NPZ files from sonar data...")
        print(f"   Max bags to process: {max_bags or 'all'}")
        
        try:
            # Default cone parameters
            if cone_params is None:
                cone_params = {
                    "fov_deg": 120.0,
                    "rmin": 0.0,
                    "rmax": 20.0,
                    "y_zoom": 10.0,
                    "grid": sonar_utils.ConeGridSpec(img_w=900, img_h=700),
                    "flip_range": False,
                    "flip_beams": True,
                    "enhanced": True,
                    "enhance_kwargs": {
                        "scale": "db",
                        "tvg": "amplitude", 
                        "p_low": 1.0,
                        "p_high": 99.5,
                        "gamma": 0.9
                    }
                }
            
            # Get exported bag stems
            stems = list_exported_bag_stems(self.exports_dir, bag_suffix="_data")
            if not stems:
                print("⚠️  No exported bag stems found. Run CSV export first.")
                return False
            
            print(f"   Found {len(stems)} exported bag stems")
            
            # Limit bags if specified
            if max_bags is not None:
                stems = stems[:max_bags]
                print(f"   Processing first {len(stems)} bags")
            
            # Create NPZ output directory
            npz_output_dir = self.exports_dir / "outputs"
            npz_output_dir.mkdir(parents=True, exist_ok=True)
            
            successful_npz = 0
            failed_npz = 0
            
            # Process each bag stem
            for bag_stem in stems:
                print(f"   Processing: {bag_stem}")
                
                # Find sonar CSV files
                csv_files = self._find_sonar_csv_files(bag_stem)
                
                if csv_files:
                    csv_path = csv_files[0]
                    try:
                        # Load sonar data
                        sonar_df = sonar_utils.load_df(csv_path)
                        print(f"     Loaded {len(sonar_df)} sonar frames")
                        
                        # Create NPZ output path
                        npz_path = npz_output_dir / f"{bag_stem}_cones.npz"
                        
                        # Generate cone data
                        result = sonar_utils.save_cone_run_npz(
                            sonar_df,
                            npz_path,
                            **cone_params,
                            progress=False  # Keep output clean
                        )
                        
                        print(f"     ✅ Created NPZ: {result['num_frames']} frames, shape {result['shape']}")
                        successful_npz += 1
                        
                    except Exception as e:
                        print(f"     ❌ Error: {e}")
                        failed_npz += 1
                else:
                    print(f"     ⚠️  No sonar CSV found")
                    failed_npz += 1
            
            self.results['npz_creation']['success'] = successful_npz > 0
            self.results['npz_creation']['files'] = successful_npz
            
            print(f"✅ NPZ creation completed!")
            print(f"   Successfully created: {successful_npz} NPZ files")
            print(f"   Failed: {failed_npz} bags")
            
            return successful_npz > 0
            
        except Exception as e:
            error_msg = f"Error during NPZ creation: {e}"
            self.results['npz_creation']['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def _find_sonar_csv_files(self, bag_stem: str) -> List[Path]:
        """Find sonar CSV files for a given bag stem."""
        search_locations = [
            self.exports_dir / "by_bag",
            self.exports_dir / "data", 
            self.exports_dir
        ]
        
        patterns = [
            f"*sonoptix*{bag_stem.replace('_data', '_video')}*.csv",
            f"*sonoptix*{bag_stem.replace('_data', '')}*.csv",
            f"sensor_sonoptix*{bag_stem.replace('_data', '_video')}*.csv",
            f"sensor_sonoptix*{bag_stem.replace('_data', '')}*.csv",
            f"{bag_stem}*sonar*.csv",
            f"{bag_stem}*ping*.csv"
        ]
        
        for search_dir in search_locations:
            if not search_dir.exists():
                continue
                
            for pattern in patterns:
                found_files = list(search_dir.glob(pattern))
                if found_files:
                    return found_files
        
        return []
    
    def generate_summary_report(self, requested_operations: List[str] = None) -> None:
        """Generate a summary report of export operations."""
        print(f"\n📋 Export Summary Report")
        print("=" * 60)
        
        # Map operation names to result keys
        operation_map = {
            'csv': 'csv_export',
            'video': 'video_export', 
            'frames': 'frame_export',
            'camera_info': 'camera_info',
            'npz': 'npz_creation'
        }
        
        total_operations = 0
        successful_operations = 0
        
        # Only show requested operations
        operations_to_show = requested_operations or ['csv']  # Default to just csv if none specified
        
        for operation in operations_to_show:
            if operation in operation_map:
                result_key = operation_map[operation]
                result = self.results[result_key]
                
                status = "✅" if result['success'] else "❌"
                print(f"{status} {operation.replace('_', ' ').title()}")
                print(f"     Files: {result['files']}")
                
                if result['errors']:
                    print(f"     Errors: {len(result['errors'])}")
                    for error in result['errors'][:2]:  # Show first 2 errors
                        print(f"       - {error}")
                    if len(result['errors']) > 2:
                        print(f"       ... and {len(result['errors']) - 2} more")
                
                total_operations += 1
                if result['success']:
                    successful_operations += 1
        
        print("\n" + "=" * 60)
        print(f"Overall: {successful_operations}/{total_operations} operations successful")
        
        # Show output directories (only those that exist and are relevant)
        print(f"\nOutput directories:")
        subdirs = [
            ("by_bag", "CSV/Parquet data files"),
            ("videos", "MP4 video files"),
            ("frames", "PNG frame sequences"),
            ("camera_info", "Camera calibration YAML"),
            ("outputs", "Cone NPZ files")
        ]
        
        for subdir, description in subdirs:
            path = self.exports_dir / subdir
            if path.exists():
                files = list(path.rglob("*"))
                # Only count actual files, not directories
                file_count = len([f for f in files if f.is_file()])
                if file_count > 0:
                    print(f"   📁 {subdir}/: {description} ({file_count} files)")


def main():
    """Main entry point for the complete export tool."""
    parser = argparse.ArgumentParser(
        description="SOLAQUA Complete Export Tool - CSV, Video, Frames, and NPZ in one script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export everything
  python solaqua_export.py --data-dir data --exports-dir exports --all

  # Just CSV export
  python solaqua_export.py --data-dir data --exports-dir exports --csv-only

  # Just video export
  python solaqua_export.py --data-dir data --exports-dir exports --video-only
  
  # CSV + quick frame sampling
  python solaqua_export.py --data-dir data --exports-dir exports --csv --frames --frame-stride 30

  # Custom video settings
  python solaqua_export.py --data-dir data --exports-dir exports --video --fps 10 --limit-videos 5
        """
    )
    
    # Directories
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing *.bag files (default: data)")
    parser.add_argument("--exports-dir", type=str, default="exports",
                       help="Directory for exports (default: exports)")
    
    # Operation modes
    parser.add_argument("--all", action="store_true",
                       help="Run all export operations")
    parser.add_argument("--csv-only", action="store_true",
                       help="Only export CSV data")
    parser.add_argument("--video-only", action="store_true",
                       help="Only export videos")
    parser.add_argument("--inspect-only", action="store_true",
                       help="Only inspect bags without exporting")
    
    # Individual operations
    parser.add_argument("--csv", action="store_true",
                       help="Export CSV data")
    parser.add_argument("--video", action="store_true", 
                       help="Export videos")
    parser.add_argument("--frames", action="store_true",
                       help="Export frame sequences")
    parser.add_argument("--camera-info", action="store_true",
                       help="Export camera info YAML")
    parser.add_argument("--npz", action="store_true",
                       help="Create cone NPZ files")
    
    # CSV options
    parser.add_argument("--file-format", choices=["csv", "parquet"], default="csv",
                       help="Export format for data (default: csv)")
    parser.add_argument("--no-video-sonar", action="store_true",
                       help="Exclude sonar topics from video bags")
    
    # Video options
    parser.add_argument("--fps", type=float,
                       help="Target FPS for videos (default: auto-detect)")
    parser.add_argument("--limit-videos", type=int,
                       help="Limit number of videos per bag")
    
    # Frame options
    parser.add_argument("--frame-stride", type=int, default=10,
                       help="Take every Nth frame (default: 10)")
    parser.add_argument("--limit-frames", type=int, default=100,
                       help="Max frames per bag (default: 100)")
    parser.add_argument("--resize", type=str,
                       help="Resize frames to WIDTHxHEIGHT (e.g., 640x480)")
    
    # NPZ options
    parser.add_argument("--max-npz-bags", type=int,
                       help="Maximum bags to process for NPZ creation")
    
    args = parser.parse_args()
    
    # Parse resize option
    resize_to = None
    if args.resize:
        try:
            w, h = map(int, args.resize.split('x'))
            resize_to = (w, h)
        except:
            print(f"❌ Invalid resize format: {args.resize}. Use WIDTHxHEIGHT")
            return 1
    
    # Initialize exporter
    exporter = SOLAQUACompleteExporter(args.data_dir, args.exports_dir)
    
    # Discover bags
    bags = exporter.discover_bags()
    if not bags['data_bags'] and not bags['video_bags']:
        print("❌ No bag files found. Check your data directory.")
        return 1
    
    # Inspect bags
    exporter.inspect_bags(bags)
    
    if args.inspect_only:
        return 0
    
    # Determine operations to run
    operations = []
    
    if args.all:
        operations = ['csv', 'video', 'frames', 'camera_info', 'npz']
    elif args.csv_only:
        operations = ['csv']
    elif args.video_only:
        operations = ['video', 'camera_info']
    else:
        if args.csv: operations.append('csv')
        if args.video: operations.append('video')
        if args.frames: operations.append('frames')
        if args.camera_info: operations.append('camera_info')
        if args.npz: operations.append('npz')
    
    # Default to CSV if nothing specified
    if not operations:
        operations = ['csv']
        print("ℹ️  No operations specified, defaulting to CSV export")
    
    print(f"\n🚀 Starting export operations: {', '.join(operations)}")
    
    # Run operations
    try:
        if 'csv' in operations:
            exporter.export_csv_data(
                file_format=args.file_format,
                include_video_sonar=not args.no_video_sonar
            )
        
        if 'video' in operations:
            exporter.export_videos(
                target_fps=args.fps,
                limit_per_bag=args.limit_videos
            )
        
        if 'frames' in operations:
            exporter.export_frames(
                stride=args.frame_stride,
                limit_per_bag=args.limit_frames,
                resize_to=resize_to
            )
        
        if 'camera_info' in operations:
            exporter.export_camera_info()
        
        if 'npz' in operations:
            exporter.create_cone_npz(max_bags=args.max_npz_bags)
        
        # Generate summary report
        exporter.generate_summary_report(operations)
        
        print(f"\n🎉 Export operations completed!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Export interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())