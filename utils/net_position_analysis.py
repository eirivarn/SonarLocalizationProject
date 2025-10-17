"""
Net Position Analysis Utilities

This module provides functions for analyzing the relationship between:
1. Net position as detected by sonar (from robot's perspective)
2. Robot position relative to the net (from navigation data)

Key concepts:
- Net position from robot: sonar-detected distance and angle to net
- Robot position to net: navigation-based position relative to net location
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS


def extract_frames_from_specific_bag(target_bag: str, exports_folder: Path, 
                                    frame_stride: int = 2, limit_frames: int = 1000) -> bool:
    """
    Extract frames directly from MP4 files for a specific bag with proper timestamps.
    
    Args:
        target_bag: Target bag name (e.g., '2024-08-20_17-02-00')
        exports_folder: Exports folder path
        frame_stride: Extract every Nth frame (default: 2)
        limit_frames: Maximum frames to extract per video (default: 1000)
        
    Returns:
        True if extraction was successful, False otherwise
    """
    import cv2
    import pandas as pd
    import os
    from utils.sonar_config import EXPORTS_SUBDIRS
    
    videos_dir = exports_folder / EXPORTS_SUBDIRS.get('videos', 'videos')
    frames_dir = exports_folder / EXPORTS_SUBDIRS.get('frames', 'frames')
    by_bag_dir = exports_folder / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    
    # Find MP4 files for the specific bag (exclude hidden files)
    target_mp4_files = [f for f in videos_dir.glob(f"*{target_bag}*compressed_image*.mp4") 
                       if not f.name.startswith('._')]
    
    if not target_mp4_files:
        print(f"No MP4 files found for bag: {target_bag}")
        return False
    
    print(f"Found {len(target_mp4_files)} MP4 file(s) for {target_bag}")
    
    success_count = 0
    for mp4_file in target_mp4_files:
        try:
            # Create output directory
            output_dir = frames_dir / f"{mp4_file.stem}_frames"
            
            # Remove hidden files that might interfere
            hidden_file = frames_dir / f"._{mp4_file.stem}_frames"
            if hidden_file.exists():
                hidden_file.unlink()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load bag CSV for timestamp conversion
            csv_pattern = f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
            csv_file = by_bag_dir / csv_pattern
            
            bag_start_time = None
            if csv_file.exists():
                try:
                    bag_df = pd.read_csv(csv_file)
                    if "ts_utc" in bag_df.columns:
                        bag_df["ts_utc"] = pd.to_datetime(bag_df["ts_utc"], utc=True)
                        bag_start_time = bag_df["ts_utc"].min()
                    elif "t" in bag_df.columns:
                        bag_df["ts_utc"] = pd.to_datetime(bag_df["t"], unit="s", utc=True)
                        bag_start_time = bag_df["ts_utc"].min()
                except Exception:
                    pass
            
            # Extract frames
            cap = cv2.VideoCapture(str(mp4_file))
            if not cap.isOpened():
                print(f"Could not open video: {mp4_file.name}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            extracted_count = 0
            index_rows = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_stride == 0 and extracted_count < limit_frames:
                    timestamp_sec = frame_count / fps if fps > 0 else frame_count
                    frame_filename = f"frame_{frame_count:06d}_{timestamp_sec:.3f}s.png"
                    frame_path = output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    extracted_count += 1
                    
                    index_entry = {
                        "video": mp4_file.name,
                        "frame_number": frame_count,
                        "timestamp_sec": timestamp_sec,
                        "file": frame_filename,
                        "extracted_count": extracted_count
                    }
                    
                    if bag_start_time is not None:
                        absolute_timestamp = bag_start_time + pd.Timedelta(seconds=timestamp_sec)
                        index_entry["ts_utc"] = absolute_timestamp
                    
                    index_rows.append(index_entry)
                
                frame_count += 1
            
            cap.release()
            
            # Save index
            if index_rows:
                index_df = pd.DataFrame(index_rows)
                index_path = output_dir / "index.csv"
                index_df.to_csv(index_path, index=False)
                
                has_absolute_ts = "ts_utc" in index_df.columns
                print(f"Extracted {extracted_count} frames from {mp4_file.name}")
                print(f"  Output: {output_dir.name}/")
                print(f"  Absolute timestamps: {has_absolute_ts}")
                
                success_count += 1
            
        except Exception as e:
            print(f"Error processing {mp4_file.name}: {e}")
            continue
    
    return success_count > 0


def load_net_analysis_results(target_bag: str, exports_folder: Path) -> pd.DataFrame:
    """
    Load net analysis results, creating them automatically if they don't exist.
    
    Args:
        target_bag: Target bag name
        exports_folder: Exports folder path
        
    Returns:
        DataFrame with net analysis results
        
    Raises:
        FileNotFoundError: If no analysis results can be found or created
    """
    outputs_dir = exports_folder / "outputs"
    
    # Try multiple possible patterns for existing analysis files
    possible_patterns = [
        f"{target_bag}_analysis_results.csv",
        f"*{target_bag}*analysis_results.csv",
        f"*{target_bag}*_cones_analysis_results.csv",
    ]
    
    # Look for existing analysis files
    for pattern in possible_patterns:
        matching_files = list(outputs_dir.glob(pattern))
        if matching_files:
            sonar_csv_path = matching_files[0]
            df_sonar = pd.read_csv(sonar_csv_path)
            print(f"Loaded analysis file: {sonar_csv_path.name} ({len(df_sonar)} frames)")
            return df_sonar
    
    # No existing analysis found - try to create it
    print(f"No analysis CSV found for {target_bag}, attempting to create from NPZ")
    
    # Check if NPZ file exists
    npz_file = outputs_dir / f"{target_bag}_cones.npz"
    if not npz_file.exists():
        # Try alternative NPZ naming patterns
        npz_patterns = [
            f"*{target_bag}*_cones.npz",
            f"{target_bag}*.npz"
        ]
        
        for pattern in npz_patterns:
            matching_npz = list(outputs_dir.glob(pattern))
            if matching_npz:
                npz_file = matching_npz[0]
                break
    
    if npz_file.exists():
        print(f"Found NPZ file: {npz_file.name}")
        return _create_analysis_from_npz(npz_file, target_bag)
    else:
        # List available files for guidance
        analysis_files = list(outputs_dir.glob("*analysis_results.csv"))
        if analysis_files:
            print(f"Available analysis files:")
            for f in analysis_files[:3]:
                print(f"  {f.name}")
        
        raise FileNotFoundError(
            f"No analysis results found for '{target_bag}'. "
            f"Create NPZ file first or provide existing analysis CSV."
        )


def _create_analysis_from_npz(npz_file: Path, target_bag: str) -> pd.DataFrame:
    """
    Create analysis results from NPZ file using DistanceAnalysisEngine.
    """
    print("Creating analysis from NPZ using DistanceAnalysisEngine...")
    
    try:
        # Try different possible import paths
        engine = None
        import_attempts = [
            "utils.image_analysis_utils",
            "image_analysis_utils", 
            "utils.distance_analysis",
            "distance_analysis"
        ]
        
        for import_path in import_attempts:
            try:
                iau = __import__(import_path, fromlist=['DistanceAnalysisEngine'])
                if hasattr(iau, 'DistanceAnalysisEngine'):
                    engine = iau.DistanceAnalysisEngine()
                    break
            except ImportError:
                continue
        
        if engine is None:
            # Search in utils directory
            try:
                import sys
                import importlib
                utils_dir = Path(__file__).parent
                for py_file in utils_dir.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    
                    module_name = f"utils.{py_file.stem}"
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, 'DistanceAnalysisEngine'):
                            engine = module.DistanceAnalysisEngine()
                            break
                    except ImportError:
                        continue
            except Exception:
                pass
        
        if engine is None:
            raise ImportError("DistanceAnalysisEngine not found. Run analysis manually in notebook 06.")
        
        # Create NPZ file index
        npz_file_index = pd.DataFrame([{
            'npz_file': npz_file,
            'bag_name': target_bag,
            'num_frames': None
        }])
        
        print("Running distance analysis...")
        net_analysis_results = engine.analyze_npz_sequence(
            npz_file_index=npz_file_index,    
            frame_start=1,
            frame_count=1500,
            frame_step=1,
            save_outputs=True
        )
        
        print(f"Analysis complete: {len(net_analysis_results)} frames")
        detection_rate = net_analysis_results['detection_success'].mean()
        print(f"Detection success rate: {detection_rate:.1%}")
        
        return net_analysis_results
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run analysis manually using notebook 06 with save_outputs=True")
        raise
    except Exception as e:
        print(f"Analysis error: {e}")
        raise


def load_navigation_data(target_bag: str, exports_folder: Optional[Path] = None) -> pd.DataFrame:
    """
    Load navigation data for the target bag.

    Args:
        target_bag: Bag name identifier
        exports_folder: Path to exports folder (default: EXPORTS_DIR_DEFAULT)

    Returns:
        DataFrame with navigation data
    """
    if exports_folder is None:
        exports_folder = Path(EXPORTS_DIR_DEFAULT)

    # Try multiple possible navigation file patterns
    possible_patterns = [
        f"sensor_navigation__{target_bag}.csv",
        f"navigation_plane_approximation__{target_bag}_data.csv",
        f"*navigation*{target_bag}*.csv"
    ]
    
    by_bag_dir = exports_folder / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    
    nav_file = None
    for pattern in possible_patterns:
        matching_files = list(by_bag_dir.glob(pattern))
        if matching_files:
            nav_file = matching_files[0]
            break
    
    if nav_file is None:
        raise FileNotFoundError(f"Navigation file not found for bag {target_bag} in {by_bag_dir}")

    print(f"Loading navigation data: {nav_file}")
    df = pd.read_csv(nav_file)
    
    # Handle different timestamp column names
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'ts_utc' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts_utc'])
    elif 't' in df.columns:
        df['timestamp'] = pd.to_datetime(df['t'], unit='s', utc=True)
    else:
        raise ValueError("No recognized timestamp column found in navigation data")

    return df


def calculate_robot_to_net_position(net_analysis_df: pd.DataFrame,
                                   navigation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate robot position relative to net using sonar detections and navigation.

    Args:
        net_analysis_df: Net analysis results with distance_pixels, angle_degrees
        navigation_df: Navigation data with position information

    Returns:
        DataFrame with combined analysis
    """
    # Create working copies
    sonar_df = net_analysis_df.copy()
    nav_df = navigation_df.copy()
    
    # Ensure both dataframes have proper timestamp columns
    if 'timestamp' not in sonar_df.columns:
        raise ValueError("Sonar dataframe missing 'timestamp' column")
    if 'timestamp' not in nav_df.columns:
        raise ValueError("Navigation dataframe missing 'timestamp' column")
    
    # Normalize timestamps to UTC and remove timezone info for merge compatibility
    sonar_df['timestamp'] = pd.to_datetime(sonar_df['timestamp'], utc=True).dt.tz_localize(None)
    nav_df['timestamp'] = pd.to_datetime(nav_df['timestamp'], utc=True).dt.tz_localize(None)
    
    # Sort both dataframes by timestamp
    sonar_df = sonar_df.sort_values('timestamp').reset_index(drop=True)
    nav_df = nav_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Timestamp range - Sonar: {sonar_df['timestamp'].min()} to {sonar_df['timestamp'].max()}")
    print(f"Timestamp range - Nav:   {nav_df['timestamp'].min()} to {nav_df['timestamp'].max()}")
    
    # Merge dataframes on timestamp (find closest navigation data for each sonar frame)
    merged_df = pd.merge_asof(
        sonar_df,
        nav_df,
        on='timestamp',
        direction='nearest',
        suffixes=('_sonar', '_nav')
    )
    
    print(f"Merged {len(merged_df)} records from {len(sonar_df)} sonar and {len(nav_df)} nav records")

    # Calculate net position in world coordinates
    # Assuming sonar angle is relative to robot heading
    # This is a simplified calculation - you may need to adjust based on your coordinate system

    results = []

    for _, row in merged_df.iterrows():
        # Check for valid detection and required navigation data
        if not row.get('detection_success', False):
            continue
            
        distance_m = row.get('distance_meters')
        angle_deg = row.get('angle_degrees')
        
        if pd.isna(distance_m) or pd.isna(angle_deg):
            continue

        # Get robot position and orientation from navigation
        # Handle different possible column names from DVL data
        robot_x = row.get('NetX', row.get('x', row.get('X', 0)))
        robot_y = row.get('NetY', row.get('y', row.get('Y', 0)))
        robot_heading = row.get('NetHeading', row.get('heading', row.get('Heading', 0)))  # Robot heading in degrees
        
        # Convert to float to avoid issues
        try:
            robot_x = float(robot_x) if not pd.isna(robot_x) else 0.0
            robot_y = float(robot_y) if not pd.isna(robot_y) else 0.0
            robot_heading = float(robot_heading) if not pd.isna(robot_heading) else 0.0
            distance_m = float(distance_m)
            angle_deg = float(angle_deg)
        except (ValueError, TypeError):
            continue

        # Convert sonar angle to world coordinates
        # Assuming angle_deg is relative to sonar centerline
        # You may need to adjust this based on your sonar mounting
        sonar_angle_world = robot_heading + angle_deg

        # Calculate net position in world coordinates
        net_x = robot_x + distance_m * np.cos(np.radians(sonar_angle_world))
        net_y = robot_y + distance_m * np.sin(np.radians(sonar_angle_world))

        results.append({
            'timestamp': row['timestamp'],
            'frame_index': row.get('frame_index', 0),
            'detection_success': True,
            'sonar_distance_m': distance_m,
            'sonar_angle_deg': angle_deg,
            'robot_x': robot_x,
            'robot_y': robot_y,
            'robot_heading': robot_heading,
            'net_x_world': net_x,
            'net_y_world': net_y,
            'sonar_angle_world': sonar_angle_world
        })

    if not results:
        print("Warning: No valid position calculations could be performed")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'timestamp', 'frame_index', 'detection_success', 'sonar_distance_m', 
            'sonar_angle_deg', 'robot_x', 'robot_y', 'robot_heading', 
            'net_x_world', 'net_y_world', 'sonar_angle_world'
        ])

    return pd.DataFrame(results)


def analyze_net_position_consistency(combined_df: pd.DataFrame) -> Dict:
    """
    Analyze consistency of net position estimates across frames.

    Args:
        combined_df: Combined net position analysis DataFrame

    Returns:
        Dictionary with consistency statistics
    """
    if len(combined_df) == 0:
        return {'error': 'No valid data for analysis'}

    # Calculate net position statistics
    net_positions = combined_df[['net_x_world', 'net_y_world']].dropna()

    if len(net_positions) == 0:
        return {'error': 'No valid net positions found'}

    # Calculate centroid and spread
    centroid_x = net_positions['net_x_world'].mean()
    centroid_y = net_positions['net_y_world'].mean()

    # Calculate distances from centroid
    distances_from_centroid = np.sqrt(
        (net_positions['net_x_world'] - centroid_x)**2 +
        (net_positions['net_y_world'] - centroid_y)**2
    )

    stats = {
        'total_frames': len(combined_df),
        'successful_detections': combined_df['detection_success'].sum(),
        'detection_rate': combined_df['detection_success'].mean() * 100,
        'net_centroid_x': centroid_x,
        'net_centroid_y': centroid_y,
        'position_std_m': distances_from_centroid.std(),
        'position_95th_percentile_m': np.percentile(distances_from_centroid, 95),
        'sonar_distance_mean': combined_df['sonar_distance_m'].mean(),
        'sonar_distance_std': combined_df['sonar_distance_m'].std(),
        'sonar_angle_std': combined_df['sonar_angle_deg'].std()
    }

    return stats


def create_net_position_visualization(combined_df: pd.DataFrame,
                                    consistency_stats: Dict) -> go.Figure:
    """
    Create focused visualization of net position estimates over time.

    Args:
        combined_df: Combined net position analysis DataFrame
        consistency_stats: Consistency statistics

    Returns:
        Plotly figure with X and Y position plots over time
    """
    # Filter to successful detections
    success_df = combined_df[combined_df['detection_success']].copy()

    if len(success_df) == 0:
        # Create empty figure if no successful detections
        fig = go.Figure()
        fig.add_annotation(text="No successful detections found",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    # Create subplots: X position over time and Y position over time
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Net X Position Over Time (Centroid: {consistency_stats.get("net_centroid_x", 0):.2f} m)',
            f'Net Y Position Over Time (Centroid: {consistency_stats.get("net_centroid_y", 0):.2f} m)'
        ),
        shared_xaxes=True
    )

    # X position over time
    fig.add_trace(
        go.Scatter(
            x=success_df['timestamp'],
            y=success_df['net_x_world'],
            mode='lines+markers',
            name='Net X Position',
            line=dict(color='blue', width=2),
            marker=dict(size=4, color='blue'),
            showlegend=False
        ),
        row=1, col=1
    )

    # Add centroid line for X
    centroid_x = consistency_stats.get('net_centroid_x', success_df['net_x_world'].mean())
    fig.add_hline(
        y=centroid_x,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Centroid: {centroid_x:.2f} m",
        annotation_position="top right",
        row=1, col=1
    )

    # Y position over time
    fig.add_trace(
        go.Scatter(
            x=success_df['timestamp'],
            y=success_df['net_y_world'],
            mode='lines+markers',
            name='Net Y Position',
            line=dict(color='green', width=2),
            marker=dict(size=4, color='green'),
            showlegend=False
        ),
        row=2, col=1
    )

    # Add centroid line for Y
    centroid_y = consistency_stats.get('net_centroid_y', success_df['net_y_world'].mean())
    fig.add_hline(
        y=centroid_y,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Centroid: {centroid_y:.2f} m",
        annotation_position="top right",
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"Net Position Estimates Over Time<br>Detection Rate: {consistency_stats.get('detection_rate', 0):.1f}% | Consistency (STD): {consistency_stats.get('position_std_m', 0):.2f} m",
        showlegend=False
    )

    # Update axis labels
    fig.update_yaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    return fig


def run_net_position_analysis(target_bag: str,
                             exports_folder: Optional[Path] = None,
                             save_results: bool = True) -> Tuple[pd.DataFrame, Dict, go.Figure]:
    """
    Complete net position analysis pipeline.

    Args:
        target_bag: Bag name identifier
        exports_folder: Path to exports folder
        save_results: Whether to save results to CSV

    Returns:
        Tuple of (combined_df, consistency_stats, visualization_figure)
    """
    print(f"=== NET POSITION ANALYSIS FOR {target_bag} ===")

    # Load data
    net_df = load_net_analysis_results(target_bag, exports_folder)
    nav_df = load_navigation_data(target_bag, exports_folder)

    print(f"Loaded {len(net_df)} net analysis frames")
    print(f"Loaded {len(nav_df)} navigation frames")

    # Calculate robot-to-net positions
    combined_df = calculate_robot_to_net_position(net_df, nav_df)

    # Analyze consistency
    consistency_stats = analyze_net_position_consistency(combined_df)

    # Create visualization
    fig = create_net_position_visualization(combined_df, consistency_stats)

    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Detection Rate: {consistency_stats['detection_rate']:.1f}%")
    print(f"Net Centroid: ({consistency_stats['net_centroid_x']:.2f}, {consistency_stats['net_centroid_y']:.2f}) m")
    print(f"Position Consistency (STD): {consistency_stats['position_std_m']:.2f} m")
    print(f"Sonar Distance Mean: {consistency_stats['sonar_distance_mean']:.2f} m")

    # Save results if requested
    if save_results and exports_folder:
        outputs_dir = exports_folder / EXPORTS_SUBDIRS.get('outputs', 'outputs')
        outputs_dir.mkdir(parents=True, exist_ok=True)

        results_file = outputs_dir / f"{target_bag}_net_position_analysis.csv"
        combined_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")

    return combined_df, consistency_stats, fig


def synchronize_sonar_dvl_frame_by_frame(df_sonar: pd.DataFrame, df_nav: pd.DataFrame, 
                                        tolerance_seconds: float = 0.5,
                                        apply_sonar_angle_correction: bool = True,
                                        angle_correction_degrees: float = -180.0) -> pd.DataFrame:
    """
    Frame-by-frame synchronization of sonar and DVL data using exact video generation logic.
    
    Args:
        df_sonar: Sonar analysis DataFrame with timestamp, distance_meters, angle_degrees
        df_nav: Navigation DataFrame with timestamp, NetDistance, NetPitch
        tolerance_seconds: Maximum time difference for synchronization
        apply_sonar_angle_correction: Whether to apply angle correction to sonar data
        angle_correction_degrees: Degrees to add to sonar angles for coordinate alignment
        
    Returns:
        DataFrame with synchronized comparisons
    """
    # Prepare data exactly like video generation
    sonar_data = df_sonar.copy()
    nav_complete = df_nav.copy()
    
    # Convert timestamps to match video generation format
    if 'timestamp' in sonar_data.columns:
        sonar_data['ts_target'] = pd.to_datetime(sonar_data['timestamp'], utc=True, errors='coerce')
    else:
        raise ValueError("Sonar data missing timestamp column")
    
    nav_complete = nav_complete.sort_values("timestamp")
    
    position_comparisons = []
    successful_syncs = 0
    
    print(f"Synchronizing {len(sonar_data)} sonar frames with DVL data...")
    
    # Process each sonar frame (same as video generation loop)
    for idx, sonar_frame in sonar_data.iterrows():
        if not sonar_frame.get('detection_success', False):
            continue
            
        ts_target = sonar_frame['ts_target']
        
        # === DVL SYNCHRONIZATION (exact video generation logic) ===
        if nav_complete is not None and len(nav_complete) > 0:
            diffs = abs(nav_complete["timestamp"] - ts_target)
            dvl_idx = diffs.idxmin()
            min_dt = diffs.iloc[dvl_idx]
            dvl_rec = nav_complete.loc[dvl_idx]
            
            if min_dt <= pd.Timedelta(f"{tolerance_seconds}s") and "NetDistance" in dvl_rec and pd.notna(dvl_rec["NetDistance"]):
                # Extract DVL measurements (exact video generation method)
                dvl_distance = float(dvl_rec["NetDistance"])
                dvl_angle_deg = 0.0
                
                if "NetPitch" in dvl_rec and pd.notna(dvl_rec["NetPitch"]):
                    dvl_angle_deg = float(np.degrees(dvl_rec["NetPitch"]))
                
                # Extract Sonar measurements with optional angle correction
                sonar_distance = float(sonar_frame["distance_meters"])
                sonar_angle_raw = float(sonar_frame["angle_degrees"])
                
                if apply_sonar_angle_correction:
                    sonar_angle_deg = sonar_angle_raw + angle_correction_degrees
                else:
                    sonar_angle_deg = sonar_angle_raw
                
                # Calculate XY relative positions (same coordinate system as video)
                # DVL: Direct distance and pitch angle
                dvl_x = dvl_distance * np.sin(np.radians(dvl_angle_deg))
                dvl_y = dvl_distance * np.cos(np.radians(dvl_angle_deg))
                
                # Sonar: Distance along center beam and corrected net orientation angle  
                sonar_x = sonar_distance * np.sin(np.radians(sonar_angle_deg))
                sonar_y = sonar_distance * np.cos(np.radians(sonar_angle_deg))
                
                # Store synchronized position comparison
                position_comparisons.append({
                    'timestamp': ts_target,
                    'frame_index': sonar_frame.get('frame_index', idx),
                    'sync_dt_seconds': min_dt.total_seconds(),
                    # Distance measurements (validated by video)
                    'dvl_distance_m': dvl_distance,
                    'dvl_angle_deg': dvl_angle_deg,
                    'sonar_distance_m': sonar_distance,
                    'sonar_angle_raw': sonar_angle_raw,
                    'sonar_angle_corrected': sonar_angle_deg,
                    # XY relative positions
                    'dvl_net_x': dvl_x,
                    'dvl_net_y': dvl_y,
                    'sonar_net_x': sonar_x,
                    'sonar_net_y': sonar_y,
                    # Position differences
                    'position_diff_x': abs(sonar_x - dvl_x),
                    'position_diff_y': abs(sonar_y - dvl_y),
                    'position_diff_magnitude': np.sqrt((sonar_x - dvl_x)**2 + (sonar_y - dvl_y)**2)
                })
                successful_syncs += 1
    
    print(f"✓ Successfully synchronized {successful_syncs} frame pairs")
    
    if successful_syncs == 0:
        print("Warning: No successful synchronizations found")
        return pd.DataFrame()
    
    return pd.DataFrame(position_comparisons)


def analyze_position_agreement(sync_df: pd.DataFrame, 
                              distance_threshold_m: float = 0.5,
                              xy_threshold_m: float = 0.5,
                              total_threshold_m: float = 1.0) -> Dict:
    """
    Analyze position agreement between synchronized sonar and DVL measurements.
    
    Args:
        sync_df: Synchronized DataFrame from synchronize_sonar_dvl_frame_by_frame
        distance_threshold_m: Threshold for distance agreement
        xy_threshold_m: Threshold for X/Y position agreement
        total_threshold_m: Threshold for total position difference
        
    Returns:
        Dictionary with agreement statistics
    """
    if len(sync_df) == 0:
        return {'error': 'no_synchronized_data'}
    
    # Basic statistics
    stats = {
        'total_synchronized_frames': len(sync_df),
        'mean_sync_time_diff_s': sync_df['sync_dt_seconds'].mean(),
        'max_sync_time_diff_s': sync_df['sync_dt_seconds'].max(),
    }
    
    # Angle correction statistics
    stats.update({
        'sonar_angle_raw_mean': sync_df['sonar_angle_raw'].mean(),
        'sonar_angle_raw_std': sync_df['sonar_angle_raw'].std(),
        'sonar_angle_corrected_mean': sync_df['sonar_angle_corrected'].mean(),
        'sonar_angle_corrected_std': sync_df['sonar_angle_corrected'].std(),
        'dvl_angle_mean': sync_df['dvl_angle_deg'].mean(),
        'dvl_angle_std': sync_df['dvl_angle_deg'].std(),
    })
    
    # Position comparison statistics
    stats.update({
        'dvl_net_x_mean': sync_df['dvl_net_x'].mean(),
        'dvl_net_x_std': sync_df['dvl_net_x'].std(),
        'dvl_net_y_mean': sync_df['dvl_net_y'].mean(),
        'dvl_net_y_std': sync_df['dvl_net_y'].std(),
        'sonar_net_x_mean': sync_df['sonar_net_x'].mean(),
        'sonar_net_x_std': sync_df['sonar_net_x'].std(),
        'sonar_net_y_mean': sync_df['sonar_net_y'].mean(),
        'sonar_net_y_std': sync_df['sonar_net_y'].std(),
    })
    
    # Position differences
    stats.update({
        'position_diff_x_mean': sync_df['position_diff_x'].mean(),
        'position_diff_y_mean': sync_df['position_diff_y'].mean(),
        'position_diff_magnitude_mean': sync_df['position_diff_magnitude'].mean(),
        'position_diff_magnitude_max': sync_df['position_diff_magnitude'].max(),
    })
    
    # Agreement analysis
    good_x_matches = (sync_df['position_diff_x'] < xy_threshold_m).sum()
    good_y_matches = (sync_df['position_diff_y'] < xy_threshold_m).sum()
    good_total_matches = (sync_df['position_diff_magnitude'] < total_threshold_m).sum()
    
    stats.update({
        'x_agreement_count': good_x_matches,
        'x_agreement_percent': (good_x_matches / len(sync_df)) * 100,
        'y_agreement_count': good_y_matches,
        'y_agreement_percent': (good_y_matches / len(sync_df)) * 100,
        'total_agreement_count': good_total_matches,
        'total_agreement_percent': (good_total_matches / len(sync_df)) * 100,
    })
    
    return stats


def create_xy_position_comparison_plot(sync_df: pd.DataFrame, title: str = None) -> go.Figure:
    """
    Create XY position comparison visualization using Plotly.
    
    Args:
        sync_df: Synchronized DataFrame from synchronize_sonar_dvl_frame_by_frame
        title: Optional plot title
        
    Returns:
        Plotly figure object
    """
    if len(sync_df) == 0:
        # Create empty figure
        fig = go.Figure()
        fig.add_annotation(text="No synchronized data available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    from plotly.subplots import make_subplots
    
    # Create time series for plotting
    time_seconds = np.arange(len(sync_df))  # Use frame sequence as time
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("X Position Over Time", "Y Position Over Time"),
        shared_xaxes=True
    )
    
    # X Position vs Time
    fig.add_trace(
        go.Scatter(x=time_seconds, y=sync_df['sonar_net_x'],
                  mode='lines+markers', name='Sonar X', 
                  line=dict(color='red', width=2), marker=dict(size=4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_seconds, y=sync_df['dvl_net_x'],
                  mode='lines+markers', name='DVL X', 
                  line=dict(color='blue', width=2), marker=dict(size=4)),
        row=1, col=1
    )
    
    # Y Position vs Time
    fig.add_trace(
        go.Scatter(x=time_seconds, y=sync_df['sonar_net_y'],
                  mode='lines+markers', name='Sonar Y', 
                  line=dict(color='orange', width=2), marker=dict(size=4)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_seconds, y=sync_df['dvl_net_y'],
                  mode='lines+markers', name='DVL Y', 
                  line=dict(color='green', width=2), marker=dict(size=4)),
        row=2, col=1
    )
    
    # Update layout
    default_title = "Sonar vs DVL Net Position Comparison (Frame-by-Frame Synchronized)"
    fig.update_layout(
        title=title or default_title,
        height=700,
        hovermode='x unified',
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (frame sequence)", row=2, col=1)
    fig.update_yaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=2, col=1)
    
    return fig


def run_complete_position_comparison(df_sonar: pd.DataFrame, df_nav: pd.DataFrame,
                                   target_bag: str = None,
                                   tolerance_seconds: float = 0.5,
                                   save_results: bool = True,
                                   exports_folder: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict, go.Figure]:
    """
    Complete pipeline for sonar vs DVL position comparison.
    
    Args:
        df_sonar: Sonar analysis DataFrame
        df_nav: Navigation DataFrame
        target_bag: Target bag identifier for saving
        tolerance_seconds: Synchronization tolerance
        save_results: Whether to save results to files
        exports_folder: Output folder for results
        
    Returns:
        Tuple of (synchronized_data, statistics, figure)
    """
    print("=== COMPLETE POSITION COMPARISON PIPELINE ===")
    
    # Step 1: Frame-by-frame synchronization
    sync_df = synchronize_sonar_dvl_frame_by_frame(
        df_sonar, df_nav, 
        tolerance_seconds=tolerance_seconds,
        apply_sonar_angle_correction=True,
        angle_correction_degrees=-180.0  # Subtract 180 degrees
    )
    
    if len(sync_df) == 0:
        print("No synchronized data - cannot proceed with analysis")
        return pd.DataFrame(), {'error': 'no_sync_data'}, go.Figure()
    
    # Step 2: Analyze agreement
    stats = analyze_position_agreement(sync_df)
    
    # Step 3: Create visualization
    fig = create_xy_position_comparison_plot(sync_df)
    
    # Step 4: Print summary
    print(f"\n=== POSITION COMPARISON RESULTS ===")
    print(f"Synchronized frames: {stats['total_synchronized_frames']}")
    print(f"Mean sync tolerance: {stats['mean_sync_time_diff_s']:.3f}s")
    print(f"\nANGLE CORRECTION:")
    print(f"  Raw sonar angles:      {stats['sonar_angle_raw_mean']:.1f} ± {stats['sonar_angle_raw_std']:.1f}°")
    print(f"  Corrected sonar angles: {stats['sonar_angle_corrected_mean']:.1f} ± {stats['sonar_angle_corrected_std']:.1f}°")
    print(f"  DVL angles:            {stats['dvl_angle_mean']:.1f} ± {stats['dvl_angle_std']:.1f}°")
    print(f"\nPOSITION COMPARISON:")
    print(f"  DVL net position X:    {stats['dvl_net_x_mean']:.3f} ± {stats['dvl_net_x_std']:.3f} m")
    print(f"  DVL net position Y:    {stats['dvl_net_y_mean']:.3f} ± {stats['dvl_net_y_std']:.3f} m") 
    print(f"  Sonar net position X:  {stats['sonar_net_x_mean']:.3f} ± {stats['sonar_net_x_std']:.3f} m")
    print(f"  Sonar net position Y:  {stats['sonar_net_y_mean']:.3f} ± {stats['sonar_net_y_std']:.3f} m")
    print(f"\nPOSITION AGREEMENT:")
    print(f"  X position agreement (<0.5m):     {stats['x_agreement_count']}/{stats['total_synchronized_frames']} ({stats['x_agreement_percent']:.1f}%)")
    print(f"  Y position agreement (<0.5m):     {stats['y_agreement_count']}/{stats['total_synchronized_frames']} ({stats['y_agreement_percent']:.1f}%)")
    print(f"  Total position agreement (<1.0m):  {stats['total_agreement_count']}/{stats['total_synchronized_frames']} ({stats['total_agreement_percent']:.1f}%)")
    
    # Step 5: Save results
    if save_results and exports_folder and target_bag:
        outputs_dir = exports_folder / EXPORTS_SUBDIRS.get('outputs', 'outputs')
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save synchronized data
        sync_file = outputs_dir / f"{target_bag}_position_comparison_sync.csv"
        sync_df.to_csv(sync_file, index=False)
        print(f"✓ Synchronized data saved to: {sync_file}")
        
        # Save statistics
        stats_file = outputs_dir / f"{target_bag}_position_comparison_stats.json"
        import json
        with open(stats_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_stats = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in stats.items()}
            json.dump(json_stats, f, indent=2)
        print(f"✓ Statistics saved to: {stats_file}")
    
    return sync_df, stats, fig