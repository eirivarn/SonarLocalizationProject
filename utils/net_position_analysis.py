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


def load_net_analysis_results(target_bag: str, exports_folder: Optional[Path] = None) -> pd.DataFrame:
    """
    Load net analysis results from CSV file.

    Args:
        target_bag: Bag name identifier
        exports_folder: Path to exports folder (default: EXPORTS_DIR_DEFAULT)

    Returns:
        DataFrame with net analysis results
    """
    if exports_folder is None:
        exports_folder = Path(EXPORTS_DIR_DEFAULT)

    # Look for analysis results CSV
    outputs_dir = exports_folder / EXPORTS_SUBDIRS.get('outputs', 'outputs')
    pattern = f"*{target_bag}*analysis_results.csv"

    matching_files = list(outputs_dir.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(f"No analysis results CSV found for bag {target_bag} in {outputs_dir}")

    csv_file = matching_files[0]  # Take first match
    print(f"Loading net analysis results: {csv_file}")

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


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

    nav_file = (exports_folder / EXPORTS_SUBDIRS.get('by_bag', 'by_bag') /
                f"navigation_plane_approximation__{target_bag}_data.csv")

    if not nav_file.exists():
        raise FileNotFoundError(f"Navigation file not found: {nav_file}")

    print(f"Loading navigation data: {nav_file}")
    df = pd.read_csv(nav_file)
    df['timestamp'] = pd.to_datetime(df['ts_utc'])

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
    # Merge dataframes on timestamp (find closest navigation data for each sonar frame)
    merged_df = pd.merge_asof(
        net_analysis_df.sort_values('timestamp'),
        navigation_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )

    # Calculate net position in world coordinates
    # Assuming sonar angle is relative to robot heading
    # This is a simplified calculation - you may need to adjust based on your coordinate system

    results = []

    for _, row in merged_df.iterrows():
        if pd.isna(row.get('distance_meters', np.nan)) or pd.isna(row.get('angle_degrees', np.nan)):
            continue

        # Get robot position and orientation from navigation
        robot_x = row.get('x', 0)  # Adjust column names based on your navigation data
        robot_y = row.get('y', 0)
        robot_heading = row.get('heading', 0)  # Robot heading in degrees

        # Sonar detection relative to robot
        distance_m = row['distance_meters']
        angle_deg = row['angle_degrees']

        # Convert sonar angle to world coordinates
        # Assuming angle_deg is relative to sonar centerline
        # You may need to adjust this based on your sonar mounting
        sonar_angle_world = robot_heading + angle_deg

        # Calculate net position in world coordinates
        net_x = robot_x + distance_m * np.cos(np.radians(sonar_angle_world))
        net_y = robot_y + distance_m * np.sin(np.radians(sonar_angle_world))

        results.append({
            'timestamp': row['timestamp'],
            'frame_index': row['frame_index'],
            'detection_success': row['detection_success'],
            'sonar_distance_m': distance_m,
            'sonar_angle_deg': angle_deg,
            'robot_x': robot_x,
            'robot_y': robot_y,
            'robot_heading': robot_heading,
            'net_x_world': net_x,
            'net_y_world': net_y,
            'sonar_angle_world': sonar_angle_world
        })

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