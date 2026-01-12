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
from typing import Dict

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