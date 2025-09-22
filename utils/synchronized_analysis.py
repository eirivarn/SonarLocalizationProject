"""
SOLAQUA Synchronized Analysis Utilities

This module provides utilities for synchronizing sonar and net distance data
with advanced timestamp alignment and analysis capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Any

# SOLAQUA utilities
import utils.sonar_utils as sonar_utils
import utils.image_analysis_utils as iau
import utils.enhanced_net_detection as end
import utils.navigation_guidance_analysis as nav_utils

# Scientific computing
try:
    from scipy import interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SynchronizedAnalyzer:
    """Main class for synchronized sonar and net distance analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the synchronized analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config
        self.nav_analyzer = nav_utils.NavigationGuidanceAnalyzer(config['by_bag_folder'])
        self.all_sonar_data = {}
        self.all_nav_data = {}
        self.synchronized_data = None
        
        # Create output directories
        Path(config['output_folder']).mkdir(parents=True, exist_ok=True)
        self.sync_output_dir = Path(config['output_folder']) / "synchronized_data"
        self.sync_output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_sonar_data(self) -> Dict[str, Any]:
        """
        Load processed sonar cone data and extract detection results with timestamps.
        
        Returns:
            Dictionary containing loaded sonar data for all bags
        """
        print("🔊 Loading sonar data...")
        print("=" * 40)
        
        # Find all available NPZ files for sonar data
        npz_candidates = list(Path(self.config['sonar_npz_folder']).glob("*_cones.npz"))
        print(f"📂 Found {len(npz_candidates)} NPZ files:")
        for npz_file in npz_candidates:
            print(f"   • {npz_file.name}")
        
        # Load ALL available sonar datasets WITH PROPER TIMESTAMPS from CSV files
        for npz_file in npz_candidates:
            # Extract bag name from filename
            bag_name = None
            for bag in self.config['all_bags']:
                if bag in npz_file.name:
                    bag_name = bag
                    break
            
            if bag_name:
                try:
                    print(f"\n📊 Loading sonar data from {bag_name}...")
                    
                    # Load cone data from NPZ
                    cones, _, extent, meta = iau.load_cone_run_npz(npz_file)
                    
                    # Load proper timestamps from CSV file
                    csv_file = Path(self.config['by_bag_folder']) / f"sensor_sonoptix_echo_image__{bag_name}_video.csv"
                    
                    if csv_file.exists():
                        print(f"   📅 Loading timestamps from CSV: {csv_file.name}")
                        
                        # Read timestamps (only the timestamp columns to save memory)
                        timestamp_df = pd.read_csv(csv_file, usecols=['ts_utc', 'ts_oslo', 't_rel'])
                        
                        if len(timestamp_df) != len(cones):
                            print(f"   ⚠️ Frame count mismatch: NPZ has {len(cones)} frames, CSV has {len(timestamp_df)} timestamps")
                            # Use the minimum of both
                            min_frames = min(len(cones), len(timestamp_df))
                            cones = cones[:min_frames]
                            timestamp_df = timestamp_df.iloc[:min_frames]
                            print(f"   🔧 Using first {min_frames} frames/timestamps")
                        
                        # Convert to proper datetime
                        frame_timestamps = pd.to_datetime(timestamp_df['ts_utc'], utc=True)
                        
                        print(f"   ✅ Loaded proper frame timestamps from CSV")
                        print(f"   📊 Timestamp range: {frame_timestamps.min()} to {frame_timestamps.max()}")
                        
                    else:
                        print(f"   ⚠️ CSV file not found: {csv_file}")
                        print(f"   🔧 Falling back to synthetic timestamps")
                        
                        # Fallback to synthetic timestamps (old method)
                        start_time = pd.to_datetime("2024-01-01", utc=True)  # Dummy time
                        num_frames = len(cones)
                        frame_interval = pd.Timedelta(seconds=0.1)  # 10 Hz
                        frame_timestamps = pd.date_range(
                            start=start_time, 
                            periods=num_frames, 
                            freq=frame_interval
                        )
                    
                    self.all_sonar_data[bag_name] = {
                        'cones': cones,
                        'timestamps': frame_timestamps,
                        'extent': extent,
                        'meta': meta,
                        'npz_file': npz_file,
                        'start_time': frame_timestamps.iloc[0],
                        'end_time': frame_timestamps.iloc[-1],
                        'has_real_timestamps': csv_file.exists()
                    }
                    
                    duration = (frame_timestamps.iloc[-1] - frame_timestamps.iloc[0]).total_seconds()
                    print(f"   ✅ Loaded: {len(cones)} frames")
                    print(f"   🗓️ Time range: {frame_timestamps.iloc[0]} to {frame_timestamps.iloc[-1]}")
                    print(f"   ⏱️ Duration: {duration:.1f} seconds")
                    print(f"   📈 Frame rate: {len(cones) / duration:.1f} Hz")
                    
                except Exception as e:
                    print(f"   ❌ Failed to load {npz_file.name}: {e}")
        
        print(f"\n✅ Loaded sonar data from {len(self.all_sonar_data)} bags: {list(self.all_sonar_data.keys())}")
        return self.all_sonar_data
    
    def run_sonar_detection(self, bag_name: str, sample_size: int = 100) -> pd.DataFrame:
        """
        Run sonar net detection analysis on selected frames.
        
        Args:
            bag_name: Name of the bag to process
            sample_size: Number of frames to sample for detection
            
        Returns:
            DataFrame with detection results
        """
        if bag_name not in self.all_sonar_data:
            raise ValueError(f"Bag {bag_name} not found in loaded sonar data")
        
        sonar_data = self.all_sonar_data[bag_name]
        cones = sonar_data['cones']
        timestamps = sonar_data['timestamps']
        extent = sonar_data['extent']
        
        print(f"\n🎯 Running sonar net detection analysis on {bag_name}...")
        
        # Reset net tracker for fresh analysis
        end.net_tracker.reset()
        
        sonar_detections = []
        sample_indices = np.linspace(0, len(cones)-1, min(len(cones), sample_size), dtype=int)
        
        for i, frame_idx in enumerate(sample_indices):
            if i % 20 == 0:
                print(f"   Processing frame {frame_idx} ({i+1}/{len(sample_indices)})")
            
            frame = cones[frame_idx]
            timestamp = timestamps[frame_idx]
            
            # Run enhanced net detection
            result = end.detect_net_with_tracking(frame, **self.config['sonar_detection_params'])
            
            # Extract detection results
            detection_record = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'net_detected': result['net_detected'],
                'tracking_position': result['tracking_info']['position'],
                'tracking_confidence': result['tracking_info']['confidence_avg'],
                'stable_frames': result['tracking_info']['stable_frames']
            }
            
            # Add detailed net information if detected
            if result['net_detected'] and result['net_info'] is not None:
                net_info = result['net_info']
                # Convert pixel coordinates to meters
                if result['tracking_info']['position'] is not None:
                    x_px, y_px = result['tracking_info']['position']
                    x_m, y_m = iau.extent_px_to_m(extent, cones[0].shape[0], cones[0].shape[1], y_px, x_px)
                    
                    detection_record.update({
                        'net_x_m': x_m,  # Starboard distance (+ = starboard)
                        'net_y_m': y_m,  # Forward distance (+ = forward)
                        'net_distance_m_sonar': y_m,  # Primary distance metric (forward to net)
                        'net_width_px': net_info['distance'],
                        'net_orientation_deg': net_info['avg_angle'],
                        'detection_confidence': net_info['confidence']
                    })
            
            sonar_detections.append(detection_record)
        
        # Convert to DataFrame
        sonar_detection_df = pd.DataFrame(sonar_detections)
        
        # Summary of sonar detections
        detections_found = sonar_detection_df['net_detected'].sum()
        print(f"✅ Sonar detection analysis complete:")
        print(f"   🔍 Frames analyzed: {len(sonar_detection_df)}")
        print(f"   🎯 Net detections: {detections_found}")
        print(f"   📊 Detection rate: {detections_found/len(sonar_detection_df)*100:.1f}%")
        
        if detections_found > 0:
            valid_distances = sonar_detection_df.dropna(subset=['net_distance_m_sonar'])
            if len(valid_distances) > 0:
                print(f"   📏 Distance range: {valid_distances['net_distance_m_sonar'].min():.2f} to {valid_distances['net_distance_m_sonar'].max():.2f} m")
                print(f"   📊 Average distance: {valid_distances['net_distance_m_sonar'].mean():.2f} ± {valid_distances['net_distance_m_sonar'].std():.2f} m")
        
        return sonar_detection_df
    
    def load_navigation_data(self) -> Dict[str, Any]:
        """
        Load navigation and guidance system net distance measurements.
        
        Returns:
            Dictionary containing loaded navigation data for all bags
        """
        print("📏 Loading net distance data from navigation/guidance systems...")
        print("=" * 60)
        
        print("🔍 Scanning all bags for navigation data...")
        for bag in self.config['all_bags']:
            print(f"\n📊 Checking bag: {bag}")
            bag_data = {}
            
            # Try guidance data
            try:
                guidance_data = self.nav_analyzer.load_sensor_data("guidance", bag, verbose=False)
                if guidance_data is not None and len(guidance_data) > 0:
                    bag_data['guidance'] = guidance_data
                    print(f"   ✅ Guidance data: {len(guidance_data)} records")
                    print(f"   🗓️ Time range: {guidance_data['ts_oslo'].min()} to {guidance_data['ts_oslo'].max()}")
            except Exception as e:
                print(f"   ⚠️ Guidance data failed: {e}")
            
            # Try navigation plane data
            try:
                nav_plane_data = self.nav_analyzer.load_sensor_data("navigation_plane_approximation", bag, verbose=False)
                if nav_plane_data is not None and len(nav_plane_data) > 0:
                    bag_data['nav_plane'] = nav_plane_data
                    print(f"   ✅ Navigation plane data: {len(nav_plane_data)} records")
                    print(f"   🗓️ Time range: {nav_plane_data['ts_oslo'].min()} to {nav_plane_data['ts_oslo'].max()}")
            except Exception as e:
                print(f"   ⚠️ Navigation plane data failed: {e}")
            
            # Try navigation position data
            try:
                nav_position_data = self.nav_analyzer.load_sensor_data("navigation_plane_approximation_position", bag, verbose=False)
                if nav_position_data is not None and len(nav_position_data) > 0:
                    bag_data['nav_position'] = nav_position_data
                    print(f"   ✅ Navigation position data: {len(nav_position_data)} records")
            except Exception as e:
                print(f"   ⚠️ Navigation position data failed: {e}")
            
            if bag_data:
                self.all_nav_data[bag] = bag_data
                print(f"   📊 Total data sources for {bag}: {len(bag_data)}")
            else:
                print(f"   ❌ No navigation data found in {bag}")
        
        return self.all_nav_data
    
    def find_best_overlap(self) -> Tuple[str, str, str]:
        """
        Find the best overlapping sonar and navigation data.
        
        Returns:
            Tuple of (sonar_bag, nav_bag, nav_source)
        """
        print("\n🔍 Finding best temporal overlap between sonar and navigation data...")
        print("=" * 65)
        
        overlap_matrix = {}
        
        for sonar_bag, sonar_data in self.all_sonar_data.items():
            sonar_start = sonar_data['start_time']
            sonar_end = sonar_data['end_time']
            
            print(f"\n📊 Checking sonar bag: {sonar_bag}")
            print(f"   🗓️ Sonar time range: {sonar_start} to {sonar_end}")
            
            for nav_bag, nav_data in self.all_nav_data.items():
                for source_name, source_data in nav_data.items():
                    # Get navigation time range (convert to UTC for comparison)
                    nav_start = pd.to_datetime(source_data['ts_oslo'].min()).tz_convert('UTC')
                    nav_end = pd.to_datetime(source_data['ts_oslo'].max()).tz_convert('UTC')
                    
                    # Calculate overlap
                    overlap_start = max(sonar_start, nav_start)
                    overlap_end = min(sonar_end, nav_end)
                    
                    if overlap_start < overlap_end:
                        overlap_duration = (overlap_end - overlap_start).total_seconds()
                        overlap_matrix[(sonar_bag, nav_bag, source_name)] = {
                            'duration': overlap_duration,
                            'start': overlap_start,
                            'end': overlap_end,
                            'sonar_coverage': overlap_duration / (sonar_end - sonar_start).total_seconds(),
                            'nav_coverage': overlap_duration / (nav_end - nav_start).total_seconds()
                        }
                        
                        print(f"   ✅ Overlap with {nav_bag}/{source_name}: {overlap_duration:.1f}s")
                    else:
                        print(f"   ❌ No overlap with {nav_bag}/{source_name}")
        
        if not overlap_matrix:
            raise ValueError("No temporal overlaps found between sonar and navigation data!")
        
        # Find best overlap (prioritize duration, then coverage)
        best_key = max(overlap_matrix.keys(), key=lambda k: (
            overlap_matrix[k]['duration'],
            min(overlap_matrix[k]['sonar_coverage'], overlap_matrix[k]['nav_coverage'])
        ))
        
        best_overlap = overlap_matrix[best_key]
        sonar_bag, nav_bag, nav_source = best_key
        
        print(f"\n🎯 Best overlap found:")
        print(f"   📊 Sonar: {sonar_bag}")
        print(f"   🧭 Navigation: {nav_bag}/{nav_source}")
        print(f"   ⏱️ Duration: {best_overlap['duration']:.1f} seconds")
        print(f"   📈 Coverage: {best_overlap['sonar_coverage']*100:.1f}% sonar, {best_overlap['nav_coverage']*100:.1f}% nav")
        
        return sonar_bag, nav_bag, nav_source
    
    def synchronize_data(self, sonar_bag: str, nav_bag: str, nav_source: str) -> pd.DataFrame:
        """
        Synchronize sonar and navigation data using timestamp alignment.
        
        Args:
            sonar_bag: Sonar bag name
            nav_bag: Navigation bag name
            nav_source: Navigation data source name
            
        Returns:
            Synchronized DataFrame
        """
        print(f"\n⏰ Synchronizing {sonar_bag} sonar with {nav_bag}/{nav_source} navigation...")
        print("=" * 70)
        
        # Get sonar detection data
        sonar_detection_df = self.run_sonar_detection(sonar_bag)
        
        # Get navigation data
        nav_data = self.all_nav_data[nav_bag][nav_source]
        
        # Prepare navigation timestamps
        nav_data = nav_data.copy()
        nav_data['timestamp'] = pd.to_datetime(nav_data['ts_oslo']).dt.tz_convert('UTC')
        
        # Extract net distance columns
        distance_cols = [col for col in nav_data.columns if 'net_distance_m' in col and col != 'net_distance_m_sonar']
        
        if distance_cols:
            # Combine multiple distance measurements if available
            nav_data['net_distance_m_net'] = nav_data[distance_cols[0]]
            print(f"   📏 Using navigation distance from: {distance_cols[0]}")
        else:
            # Check for alternative distance column names
            alt_distance_cols = [col for col in nav_data.columns if 'distance' in col.lower()]
            if alt_distance_cols:
                nav_data['net_distance_m_net'] = nav_data[alt_distance_cols[0]]
                print(f"   📏 Using alternative distance from: {alt_distance_cols[0]}")
            else:
                # Fallback to a default value if no distance measurements
                nav_data['net_distance_m_net'] = 0.5  # Default 0.5m distance
                print(f"   ⚠️ No distance measurements found, using default value")
        
        # Add source identifier if not present
        if 'source' not in nav_data.columns:
            nav_data['source'] = f"{nav_bag}_{nav_source}"
        
        # Perform timestamp synchronization using merge_asof
        print(f"   🔄 Performing timestamp synchronization...")
        print(f"   ⏰ Time tolerance: {self.config['time_tolerance']}")
        
        # Sort both datasets by timestamp
        sonar_sync = sonar_detection_df.sort_values('timestamp').reset_index(drop=True)
        nav_sync = nav_data.sort_values('timestamp').reset_index(drop=True)
        
        # Use pandas merge_asof for nearest timestamp matching
        # Select only available columns for merging
        merge_cols = ['timestamp', 'net_distance_m_net']
        available_extra_cols = ['t', 'ts_oslo', 't_rel', 'source', 't_rel_sec']
        for col in available_extra_cols:
            if col in nav_sync.columns:
                merge_cols.append(col)
        
        synchronized_data = pd.merge_asof(
            sonar_sync,
            nav_sync[merge_cols],
            on='timestamp',
            tolerance=self.config['time_tolerance'],
            direction='nearest'
        )
        
        # Filter out records without matches
        matched_data = synchronized_data.dropna(subset=['net_distance_m_net']).copy()
        
        print(f"   ✅ Synchronization complete!")
        print(f"   📊 Original sonar records: {len(sonar_sync)}")
        print(f"   🧭 Original nav records: {len(nav_sync)}")
        print(f"   🎯 Synchronized records: {len(matched_data)}")
        print(f"   📈 Match rate: {len(matched_data)/len(sonar_sync)*100:.1f}%")
        
        if len(matched_data) > 0:
            # Calculate time differences
            time_diffs = pd.to_datetime(matched_data['timestamp']) - pd.to_datetime(matched_data['ts_oslo'])
            time_diffs_ms = time_diffs.dt.total_seconds() * 1000
            
            print(f"   ⏱️ Time alignment quality:")
            print(f"      Mean difference: {time_diffs_ms.mean():.1f} ms")
            print(f"      Std difference: {time_diffs_ms.std():.1f} ms")
            print(f"      Max difference: {abs(time_diffs_ms).max():.1f} ms")
        
        self.synchronized_data = matched_data
        return matched_data
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the quality and consistency of synchronized data.
        
        Args:
            data: Synchronized DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        print("Validating synchronized data quality and consistency...")
        print("=" * 55)
        
        # Temporal validation
        print("Temporal validation:")
        print("-" * 25)
        
        time_deltas = data['timestamp'].diff().dt.total_seconds()
        time_deltas = time_deltas.dropna()
        
        temporal_metrics = {}
        if len(time_deltas) > 0:
            temporal_metrics = {
                'mean_interval': float(time_deltas.mean()),
                'std_interval': float(time_deltas.std()),
                'min_interval': float(time_deltas.min()),
                'max_interval': float(time_deltas.max())
            }
            
            print(f"  Time intervals - Mean: {temporal_metrics['mean_interval']:.2f}s, Std: {temporal_metrics['std_interval']:.2f}s")
            print(f"  Min interval: {temporal_metrics['min_interval']:.2f}s, Max interval: {temporal_metrics['max_interval']:.2f}s")
            
            # Check for irregular intervals
            irregular_intervals = time_deltas[abs(time_deltas - time_deltas.mean()) > 2 * time_deltas.std()]
            temporal_metrics['irregular_count'] = len(irregular_intervals)
            
            if len(irregular_intervals) > 0:
                print(f"  Irregular intervals detected: {len(irregular_intervals)} instances")
            else:
                print(f"  Temporal consistency: Good")
        
        # Distance correlation analysis
        print(f"\nDistance measurement correlation:")
        print("-" * 35)
        
        correlation_metrics = {}
        if 'net_distance_m_sonar' in data.columns and 'net_distance_m_net' in data.columns:
            both_measurements = data.dropna(subset=['net_distance_m_sonar', 'net_distance_m_net'])
            
            if len(both_measurements) > 1:
                correlation = both_measurements['net_distance_m_sonar'].corr(both_measurements['net_distance_m_net'])
                distance_diff = both_measurements['net_distance_m_sonar'] - both_measurements['net_distance_m_net']
                
                correlation_metrics = {
                    'correlation': float(correlation),
                    'mean_difference': float(distance_diff.mean()),
                    'std_difference': float(distance_diff.std()),
                    'mean_abs_difference': float(abs(distance_diff).mean()),
                    'records_with_both': len(both_measurements)
                }
                
                print(f"  Records with both measurements: {len(both_measurements)}")
                print(f"  Sonar-Nav correlation: {correlation:.3f}")
                print(f"  Distance difference - Mean: {distance_diff.mean():.3f}m +/- {distance_diff.std():.3f}m")
                print(f"  Mean absolute difference: {abs(distance_diff).mean():.3f}m")
        
        # Data completeness
        completeness = {
            'total_records': len(data),
            'sonar_detection_rate': float(data['net_detected'].mean() * 100),
            'sonar_distance_completeness': float(data['net_distance_m_sonar'].notna().mean() * 100) if 'net_distance_m_sonar' in data.columns else 0,
            'nav_distance_completeness': float(data['net_distance_m_net'].notna().mean() * 100) if 'net_distance_m_net' in data.columns else 0,
        }
        
        print(f"\nOverall data quality summary:")
        print("-" * 32)
        print(f"  Total synchronized records: {completeness['total_records']}")
        print(f"  Sonar detection rate: {completeness['sonar_detection_rate']:.1f}%")
        print(f"  Data completeness rates:")
        print(f"    Sonar distance: {completeness['sonar_distance_completeness']:.1f}%")
        print(f"    Nav distance: {completeness['nav_distance_completeness']:.1f}%")
        
        overall_quality = 'Good' if min(completeness['sonar_distance_completeness'], completeness['nav_distance_completeness']) > 80 else 'Moderate'
        print(f"\nValidation complete! Data quality appears {overall_quality}")
        
        return {
            'temporal': temporal_metrics,
            'correlation': correlation_metrics,
            'completeness': completeness,
            'overall_quality': overall_quality
        }
    
    def export_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Export synchronized data to multiple formats.
        
        Args:
            data: Synchronized DataFrame to export
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with export file paths
        """
        print("Exporting synchronized sonar and net distance dataset...")
        print("=" * 55)
        
        # Create export filename
        export_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_selection = metadata.get('source_bag', 'unknown')
        export_filename = f"synchronized_sonar_net_distance_{bag_selection}_{export_timestamp}"
        
        exported_files = {}
        
        print(f"Export preparation:")
        print(f"  Dataset: {len(data)} synchronized records")
        print(f"  Quality: {metadata.get('quality_level', 'unknown')}")
        
        # Export to CSV
        if self.config.get('export_csv', True):
            csv_file = self.sync_output_dir / f"{export_filename}.csv"
            data.to_csv(csv_file, index=False)
            exported_files['csv'] = str(csv_file)
            print(f"  ✅ CSV exported: {csv_file.name}")
        
        # Export metadata to JSON
        if self.config.get('export_json', True):
            json_file = self.sync_output_dir / f"{export_filename}_metadata.json"
            
            # Prepare JSON-serializable metadata
            json_metadata = metadata.copy()
            if 'export_timestamp' in json_metadata and hasattr(json_metadata['export_timestamp'], 'isoformat'):
                json_metadata['export_timestamp'] = json_metadata['export_timestamp'].isoformat()
            
            with open(json_file, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            exported_files['json'] = str(json_file)
            print(f"  ✅ Metadata JSON exported: {json_file.name}")
        
        # Export to NPZ
        if self.config.get('export_npz', True):
            npz_file = self.sync_output_dir / f"{export_filename}.npz"
            
            # Prepare numerical arrays
            arrays_to_save = {}
            for col in data.columns:
                if data[col].dtype in ['float64', 'int64', 'bool']:
                    arrays_to_save[col] = data[col].values
                elif col == 'timestamp':
                    # Convert timestamp to Unix timestamp
                    arrays_to_save['timestamp_unix'] = data['timestamp'].astype('int64') // 10**9
            
            np.savez_compressed(npz_file, **arrays_to_save)
            exported_files['npz'] = str(npz_file)
            print(f"  ✅ NPZ exported: {npz_file.name}")
        
        print(f"\nExport complete!")
        print(f"Files saved to: {self.sync_output_dir}")
        
        return exported_files


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for synchronized analysis."""
    return {
        'by_bag_folder': "exports/by_bag",
        'output_folder': "exports/outputs",
        'sonar_npz_folder': "exports/outputs",
        'all_bags': [
            "2024-08-20_13-39-34",
            "2024-08-20_13-40-35", 
            "2024-08-22_14-06-43",
            "2024-08-22_14-29-05",
            "2024-08-22_14-47-39"
        ],
        'time_tolerance': pd.Timedelta("100ms"),
        'time_interpolation': True,
        'max_gap_fill': pd.Timedelta("500ms"),
        'sonar_detection_params': {
            'blur_ksize': 70,
            'blur_sigma': 10.0, 
            'thr_percentile': 80,
            'min_area_px': 200,
            'verbose': False
        },
        'export_csv': True,
        'export_npz': True,
        'export_json': True,
        'export_plots': True
    }


def run_full_analysis(config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the complete synchronized analysis pipeline.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (synchronized_data, quality_metrics)
    """
    if config is None:
        config = create_default_config()
    
    analyzer = SynchronizedAnalyzer(config)
    
    # Load data
    analyzer.load_sonar_data()
    analyzer.load_navigation_data()
    
    # Find best overlap
    sonar_bag, nav_bag, nav_source = analyzer.find_best_overlap()
    
    # Synchronize data
    synchronized_data = analyzer.synchronize_data(sonar_bag, nav_bag, nav_source)
    
    # Analyze quality
    quality_metrics = analyzer.analyze_data_quality(synchronized_data)
    
    # Prepare export metadata
    metadata = {
        'total_records': len(synchronized_data),
        'source_bag': sonar_bag,
        'nav_bag': nav_bag,
        'nav_source': nav_source,
        'quality_level': quality_metrics['overall_quality'].lower(),
        'export_timestamp': datetime.now(),
        'sync_method': 'merge_asof',
        'time_tolerance_ms': config['time_tolerance'].total_seconds() * 1000,
        'quality_metrics': quality_metrics
    }
    
    # Export data
    exported_files = analyzer.export_data(synchronized_data, metadata)
    
    print(f"\n🎯 Analysis complete!")
    print(f"📊 Synchronized records: {len(synchronized_data)}")
    print(f"🏆 Overall quality: {quality_metrics['overall_quality']}")
    print(f"📁 Files exported: {list(exported_files.keys())}")
    
    return synchronized_data, quality_metrics
