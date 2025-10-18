# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple, Optional

def load_all_distance_data_for_bag(target_bag: str, exports_folder: str | None = None) -> Tuple[Dict, Dict]:
    """
    Load all distance measurement data for a specific bag.
    
    Args:
        target_bag: Bag identifier (e.g., "2024-08-22_14-29-05")
        exports_folder: Path to exports folder
        
    Returns:
        Tuple of (raw_data_dict, distance_measurements_dict)
    """
    
    print(f" LOADING ALL DISTANCE DATA FOR BAG: {target_bag}")
    print("=" * 60)
    
    from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    if exports_folder is None:
        data_folder = Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    else:
        # If exports_folder is provided, check if it's the root exports dir or by_bag dir
        ef_path = Path(exports_folder)
        if ef_path.name == 'by_bag' or (ef_path / 'by_bag').exists():
            # User passed the exports root, append by_bag
            data_folder = ef_path if ef_path.name == 'by_bag' else ef_path / 'by_bag'
        else:
            # User passed the full by_bag path
            data_folder = ef_path
            
    distance_measurements = {}
    raw_data = {}
    
    # 1. Load Navigation Data
    print(" 1. Loading Navigation Data...")
    try:
        nav_file = data_folder / f"navigation_plane_approximation__{target_bag}_data.csv"
        if nav_file.exists():
            # First try to read all columns to see what's available
            nav_data_full = pd.read_csv(nav_file)
            available_cols = nav_data_full.columns.tolist()
            
            # Required columns
            required_cols = ['ts_oslo', 'NetDistance', 'Altitude']
            optional_cols = ['NetPitch']
            
            cols_to_load = required_cols + [col for col in optional_cols if col in available_cols]
            
            if set(required_cols).issubset(set(available_cols)):
                nav_data = nav_data_full[cols_to_load].copy()
                nav_data['timestamp'] = pd.to_datetime(nav_data['ts_oslo']).dt.tz_convert('UTC')
                nav_data['net_distance_m_raw'] = nav_data['NetDistance']
                
                # Include NetPitch in the final dataframe if available
                final_cols = ['timestamp', 'net_distance_m_raw', 'NetDistance', 'Altitude']
                if 'NetPitch' in nav_data.columns:
                    final_cols.append('NetPitch')
                
                raw_data['navigation'] = nav_data[final_cols].copy()
                print(f"    Loaded {len(nav_data)} navigation records")
                if 'NetPitch' in nav_data.columns:
                    print(f"    NetPitch data available: {nav_data['NetPitch'].notna().sum()} valid records")
                else:
                    print(f"    NetPitch data not available in navigation file")
            else:
                print(f"    Required columns {required_cols} not found in navigation file")
                raw_data['navigation'] = None
        else:
            print(f"    Navigation file not found")
            raw_data['navigation'] = None
    except Exception as e:
        print(f"    Error loading navigation: {e}")
        raw_data['navigation'] = None

    # 2. Load Guidance Data  
    print(" 2. Loading Guidance Data...")
    try:
        guidance_file = data_folder / f"guidance__{target_bag}_data.csv"
        if guidance_file.exists():
            guidance_data = pd.read_csv(guidance_file)
            guidance_data['timestamp'] = pd.to_datetime(guidance_data['ts_oslo']).dt.tz_convert('UTC')
            distance_cols = [col for col in guidance_data.columns if 'distance' in col.lower()]
            if distance_cols:
                raw_data['guidance'] = guidance_data[['timestamp'] + distance_cols].copy()
                print(f"    Loaded {len(guidance_data)} guidance records with {distance_cols}")
            else:
                raw_data['guidance'] = None
                print(f"     No distance columns in guidance data")
        else:
            print(f"    Guidance file not found")
            raw_data['guidance'] = None
    except Exception as e:
        print(f"    Error loading guidance: {e}")
        raw_data['guidance'] = None

    # 3. Load DVL Altimeter
    print(" 3. Loading DVL Altimeter...")
    try:
        dvl_alt_file = data_folder / f"nucleus1000dvl_altimeter__{target_bag}_data.csv"
        if dvl_alt_file.exists():
            dvl_alt = pd.read_csv(dvl_alt_file)
            dvl_alt['timestamp'] = pd.to_datetime(dvl_alt['ts_utc'])
            if 'altimeter_distance' in dvl_alt.columns:
                distance_measurements['DVL_Altimeter'] = {
                    'data': dvl_alt,
                    'distance_col': 'altimeter_distance',
                    'description': 'Distance to seafloor',
                    'color': 'orange'
                }
                print(f"    Loaded {len(dvl_alt)} DVL altimeter records")
            else:
                print(f"     No altimeter_distance column")
        else:
            print(f"    DVL altimeter file not found")
    except Exception as e:
        print(f"    Error loading DVL altimeter: {e}")

    # 4. Load USBL
    print(" 4. Loading USBL...")
    try:
        usbl_file = data_folder / f"sensor_usbl__{target_bag}_data.csv"
        if usbl_file.exists():
            usbl = pd.read_csv(usbl_file)
            usbl['timestamp'] = pd.to_datetime(usbl['ts_utc'])
            if all(col in usbl.columns for col in ['east', 'north', 'depth']):
                usbl['usbl_distance'] = np.sqrt(usbl['east']**2 + usbl['north']**2 + usbl['depth']**2)
                distance_measurements['USBL_3D'] = {
                    'data': usbl,
                    'distance_col': 'usbl_distance',
                    'description': '3D acoustic position',
                    'color': 'purple'
                }
                distance_measurements['USBL_Depth'] = {
                    'data': usbl,
                    'distance_col': 'depth',
                    'description': 'USBL depth measurement',
                    'color': 'magenta'
                }
                print(f"    Loaded {len(usbl)} USBL records")
            else:
                print(f"     Missing USBL position columns")
        else:
            print(f"    USBL file not found")
    except Exception as e:
        print(f"    Error loading USBL: {e}")

    # 5. Load DVL Position
    print(" 5. Loading DVL Position...")
    try:
        dvl_pos_file = data_folder / f"sensor_dvl_position__{target_bag}_data.csv"
        if dvl_pos_file.exists():
            dvl_pos = pd.read_csv(dvl_pos_file)
            dvl_pos['timestamp'] = pd.to_datetime(dvl_pos['ts_utc'])
            if all(col in dvl_pos.columns for col in ['x', 'y', 'z']):
                dvl_pos['dvl_3d_distance'] = np.sqrt(dvl_pos['x']**2 + dvl_pos['y']**2 + dvl_pos['z']**2)
                distance_measurements['DVL_Position'] = {
                    'data': dvl_pos,
                    'distance_col': 'dvl_3d_distance',
                    'description': '3D DVL position',
                    'color': 'cyan'
                }
                print(f"    Loaded {len(dvl_pos)} DVL position records")
            else:
                print(f"     Missing DVL position columns")
        else:
            print(f"    DVL position file not found")
    except Exception as e:
        print(f"    Error loading DVL position: {e}")
    
    # 6. Load Navigation Position
    print(" 6. Loading Navigation Position...")
    try:
        nav_pos_file = data_folder / f"navigation_plane_approximation_position__{target_bag}_data.csv"
        if nav_pos_file.exists():
            nav_pos = pd.read_csv(nav_pos_file)
            nav_pos['timestamp'] = pd.to_datetime(nav_pos['ts_utc'])
            if all(col in nav_pos.columns for col in ['x', 'y']):
                nav_pos['nav_2d_distance'] = np.sqrt(nav_pos['x']**2 + nav_pos['y']**2)
                distance_measurements['Nav_Position'] = {
                    'data': nav_pos,
                    'distance_col': 'nav_2d_distance',
                    'description': '2D navigation position',
                    'color': 'brown'
                }
                print(f"    Loaded {len(nav_pos)} navigation position records")
            else:
                print(f"     Missing navigation position columns")
        else:
            print(f"    Navigation position file not found")
    except Exception as e:
        print(f"    Error loading navigation position: {e}")

    # 7. Load INS Z Position
    print(" 7. Loading INS Z Position...")
    try:
        ins_file = data_folder / f"nucleus1000dvl_ins__{target_bag}_data.csv"
        if ins_file.exists():
            ins_data = pd.read_csv(ins_file)
            ins_data['timestamp'] = pd.to_datetime(ins_data['ts_utc'])

            # Find z position column
            z_cols = [col for col in ins_data.columns if 'z' in col.lower() and ('pos' in col.lower() or 'position' in col.lower() or col.lower() == 'z')]
            depth_cols = [col for col in ins_data.columns if 'depth' in col.lower()]
            altitude_cols = [col for col in ins_data.columns if 'alt' in col.lower()]

            z_position_col = None
            if z_cols:
                z_position_col = z_cols[0]
            elif depth_cols:
                z_position_col = depth_cols[0]
            elif altitude_cols:
                z_position_col = altitude_cols[0]

            if z_position_col:
                z_values = ins_data[z_position_col].dropna()
                if len(z_values) > 0:
                    distance_measurements['INS_Z_Position'] = {
                        'data': ins_data,
                        'distance_col': z_position_col,
                        'description': f'INS {z_position_col} (vertical position)',
                        'color': 'darkblue'
                    }
                    print(f"    Loaded {len(ins_data)} INS records with {z_position_col}")
                else:
                    print(f"     No valid {z_position_col} values")
            else:
                print(f"     No Z position columns found")
        else:
            print(f"    INS file not found")
    except Exception as e:
        print(f"    Error loading INS: {e}")
    
    # Summary
    print(f"\n LOADING SUMMARY:")
    print(f"    Target bag: {target_bag}")
    print(f"    Raw data loaded: {len([k for k, v in raw_data.items() if v is not None])}/{len(raw_data)}")
    print(f"    Distance measurements: {len(distance_measurements)}")
    
    return raw_data, distance_measurements


def collect_distance_measurements_at_timestamp(sonar_timestamp, raw_nav_data, raw_guidance_data, distance_measurements, tolerance_sec=1.0):
    """
    Collect all available distance measurements synchronized to a sonar timestamp.
    
    Args:
        sonar_timestamp: Target timestamp for synchronization
        raw_nav_data: Navigation DataFrame
        raw_guidance_data: Guidance DataFrame  
        distance_measurements: Dictionary of sensor measurements
        tolerance_sec: Time synchronization tolerance in seconds
        
    Returns:
        Dictionary of synchronized distance measurements
    """
    
    distance_data = {}
    tolerance = pd.Timedelta(f'{tolerance_sec}s')
    
    # 1. Navigation NetDistance (primary)
    if raw_nav_data is not None:
        time_diffs = abs(raw_nav_data['timestamp'] - sonar_timestamp)
        nav_idx = time_diffs.idxmin()
        if time_diffs.min() <= tolerance:
            nav_distance = raw_nav_data.loc[nav_idx, 'NetDistance']
            distance_data['Navigation NetDistance'] = {
                'value': nav_distance,
                'color': 'red',
                'style': '-',
                'width': 3,
                'description': 'Primary navigation measurement'
            }
    
    # 2. Guidance distances
    if raw_guidance_data is not None:
        guidance_time_diffs = abs(raw_guidance_data['timestamp'] - sonar_timestamp)
        if guidance_time_diffs.min() <= tolerance:
            guidance_idx = guidance_time_diffs.idxmin()
            guidance_row = raw_guidance_data.loc[guidance_idx]
            
            # Check for different guidance distance columns
            if 'desired_net_distance' in guidance_row and pd.notna(guidance_row['desired_net_distance']):
                distance_data['Desired Distance'] = {
                    'value': guidance_row['desired_net_distance'],
                    'color': 'green',
                    'style': ':',
                    'width': 2,
                    'description': 'Target/desired distance'
                }
            
            if 'error_net_distance' in guidance_row and pd.notna(guidance_row['error_net_distance']):
                distance_data['Guidance Error'] = {
                    'value': guidance_row['error_net_distance'],
                    'color': 'blue',
                    'style': '--',
                    'width': 2,
                    'description': 'Navigation error distance'
                }
    
    # 3. All other sensor measurements
    if distance_measurements:
        for dist_name, dist_info in distance_measurements.items():
            dist_data = dist_info['data']
            dist_col = dist_info['distance_col']
            
            # Find closest measurement within tolerance
            dist_time_diffs = abs(dist_data['timestamp'] - sonar_timestamp)
            if dist_time_diffs.min() <= pd.Timedelta('2s'):  # Slightly larger tolerance for other sensors
                dist_idx = dist_time_diffs.idxmin()
                dist_row = dist_data.loc[dist_idx]
                
                if dist_col in dist_row and pd.notna(dist_row[dist_col]):
                    dist_val = dist_row[dist_col]
                    
                    # Special handling for INS Z position (might be negative depth)
                    if dist_name == 'INS_Z_Position':
                        if dist_val < 0:
                            dist_val = abs(dist_val)  # Convert negative depth to positive distance
                        
                        distance_data[dist_name] = {
                            'value': dist_val,
                            'color': dist_info['color'],
                            'style': '-',
                            'width': 2,
                            'description': f"{dist_info['description']} (abs value)"
                        }
                        
                    # Regular distance measurements (positive values only)
                    elif 0.1 <= dist_val <= 20.0:
                        distance_data[dist_name] = {
                            'value': dist_val,
                            'color': dist_info['color'],
                            'style': '-.',
                            'width': 1.5,
                            'description': dist_info['description']
                        }
    
    return distance_data

def get_configurable_sonar_parameters(target_bag: str, rmax: float = None) -> Dict:
    """
    Get configurable sonar parameters for a specific bag.
    You can manually specify the rmax or let it default to 20.0m.
    
    Args:
        target_bag: Bag identifier (e.g., "2024-08-22_14-29-05")
        rmax: Manual override for maximum range (if None, defaults to 20.0m)
        
    Returns:
        Dictionary with sonar parameters
    """
    
    # Import here to avoid circular imports
    from utils.sonar_utils import ConeGridSpec
    
    # Use provided rmax or default to 20.0m
    if rmax is None:
        rmax = 20.0
        range_source = "default"
    else:
        range_source = "manual"
    
    # Determine run type from bag name for description
    if "2024-08-20" in target_bag:
        run_type = "calibration"
        description = "Stereo camera calibration runs"
    elif "2024-08-22_14-06" in target_bag:
        run_type = "multi_dvl_early"
        description = "NFH, 2m depth, 0.5-1.0m distance, 0.2 m/s speed"
    elif "2024-08-22_14-29" in target_bag or "2024-08-22_14-47" in target_bag:
        run_type = "multi_dvl_later"
        description = "NFH, 2m depth, refined distance control"
    else:
        run_type = "unknown"
        description = "Unknown experimental configuration"
    
    sonar_params = {
        'fov_deg': 120.0,
        'rmin': 0.0,
        'rmax': float(rmax),
        'y_zoom': 5.0,  # 5m visualization range
        'grid': ConeGridSpec(img_w=900, img_h=700),
        'enhanced': True,
        'run_type': run_type,
        'description': description,
        'range_source': range_source
    }
    
    return sonar_params

def extract_raw_sonar_data_with_configurable_rmax(
    target_bag: str,
    frame_idx: int,
    rmax: float = None,
    exports_folder: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict, Dict]:
    """
    Extract raw sonar data and process it with a configurable rmax setting.
    
    Args:
        target_bag: Bag identifier (e.g., "2024-08-22_14-29-05")
        frame_idx: Frame index to extract
        rmax: Maximum range in meters (if None, defaults to 20.0m)
        exports_folder: Path to exports folder
        
    Returns:
        Tuple of (raw_sonar_matrix, processed_cone, extent, sonar_params)
    """
    
    import utils.sonar_utils as sonar_utils
    
    print(f" EXTRACTING RAW SONAR DATA WITH CONFIGURABLE RMAX")
    print(f"    Bag: {target_bag}")
    print(f"    Frame: {frame_idx}")
    print(f"    rmax: {rmax if rmax else 'default (20.0m)'}")
    print("=" * 50)
    
    # 1. Get configurable sonar parameters
    sonar_params = get_configurable_sonar_parameters(target_bag, rmax)
    
    print(f" SONAR PARAMETERS:")
    print(f"    Run Type: {sonar_params['run_type']}")
    print(f"    Range Max: {sonar_params['rmax']}m ({sonar_params['range_source']})")
    print(f"    Description: {sonar_params['description']}")
    
    # 2. Load raw sonar data
    from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    exports_root = Path(exports_folder) if exports_folder is not None else Path(EXPORTS_DIR_DEFAULT)
    sonar_csv_file = exports_root / EXPORTS_SUBDIRS.get('by_bag', 'by_bag') / f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
    
    if not sonar_csv_file.exists():
        print(f" Sonar data file not found: {sonar_csv_file}")
        return None, None, {}, sonar_params
    
    try:
        print(f"📡 Loading sonar data from: {sonar_csv_file.name}")
        sonar_df = pd.read_csv(sonar_csv_file)
        print(f"    Loaded {len(sonar_df)} sonar frames")
    except Exception as e:
        print(f" Error loading sonar data: {e}")
        return None, None, {}, sonar_params
    
    # 3. Validate frame index
    if frame_idx >= len(sonar_df):
        print(f" Frame {frame_idx} not available (only {len(sonar_df)} frames)")
        frame_idx = min(frame_idx, len(sonar_df) - 1)
        print(f"    Adjusted to frame: {frame_idx}")
    
    # 4. Extract raw sonar frame matrix
    try:
        print(f"🔍 Extracting raw sonar frame {frame_idx}...")
        raw_matrix = sonar_utils.get_sonoptix_frame(sonar_df, frame_idx)
        
        if raw_matrix is None:
            print(f" Could not extract sonar frame {frame_idx}")
            return None, None, {}, sonar_params
            
        print(f"    Raw matrix shape: {raw_matrix.shape}")
        print(f"    Value range: {raw_matrix.min():.3f} to {raw_matrix.max():.3f}")
        
    except Exception as e:
        print(f" Error extracting raw sonar frame: {e}")
        return None, None, {}, sonar_params
    
    # 5. Process with specified rmax setting
    try:
        print(f"⚙️  Processing with rmax={sonar_params['rmax']}m...")
        
        # Enhance intensity using specified rmax
        enhanced_matrix = sonar_utils.enhance_intensity(
            raw_matrix, 
            sonar_params['rmin'], 
            sonar_params['rmax']
        )
        
        # Rasterize to cone with specified parameters
        cone, extent = sonar_utils.rasterize_cone(
            enhanced_matrix,
            fov_deg=sonar_params['fov_deg'],
            rmin=sonar_params['rmin'],
            rmax=sonar_params['rmax'], 
            y_zoom=sonar_params['y_zoom'],
            grid=sonar_params['grid']
        )
        
        print(f"    Processed cone shape: {cone.shape}")
        print(f"    Extent: {extent}")
        
        # Get timestamp
        sonar_timestamp = pd.to_datetime(sonar_df.loc[frame_idx, 'ts_utc'])
        print(f"    Timestamp: {sonar_timestamp.strftime('%H:%M:%S')}")
        
        return raw_matrix, cone, extent, sonar_params
        
    except Exception as e:
        print(f" Error processing sonar data: {e}")
        return raw_matrix, None, {}, sonar_params

# Function to process sonar data and create enhanced visualization
def process_sonar_data_and_visualize(
    processed_cone, extent, sonar_params, target_bag, frame_index,
    exports_folder, nav_data, guidance_data, distance_measurements
):
    if processed_cone is None:
        print(f" Failed to extract raw sonar data for {target_bag}, frame {frame_index}")
        return

    # Resolve exports folder from config if needed and get synchronized distance measurements
    from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    exports_root = Path(exports_folder) if exports_folder is not None else Path(EXPORTS_DIR_DEFAULT)
    sonar_csv_file = exports_root / EXPORTS_SUBDIRS.get('by_bag', 'by_bag') / f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
    if not sonar_csv_file.exists():
        print(f" Sonar CSV file not found: {sonar_csv_file}")
        return

    sonar_df = pd.read_csv(sonar_csv_file)
    if frame_index >= len(sonar_df):
        print(f" Frame index {frame_index} out of range for sonar data")
        return

    sonar_timestamp = pd.to_datetime(sonar_df.loc[frame_index, 'ts_utc'])
    distance_data = collect_distance_measurements_at_timestamp(
        sonar_timestamp, nav_data, guidance_data, distance_measurements
    )

    # Enhanced net PITCH angle extraction (local orientation)
    net_angle_rad = 0.0
    net_angle_deg = 0.0
    angle_source = "default (straight ahead)"

    # Try navigation data (NetPitch) - LOCAL ORIENTATION
    if nav_data is not None and 'NetPitch' in nav_data.columns:
        nav_time_diffs = abs(nav_data['timestamp'] - sonar_timestamp)
        min_time_diff = nav_time_diffs.min()

        if min_time_diff <= pd.Timedelta('5s'):
            nav_idx = nav_time_diffs.idxmin()
            # Invert sign of NetPitch for plotting/rotation
            net_angle_rad = -nav_data.loc[nav_idx, 'NetPitch']
            net_angle_deg = np.degrees(net_angle_rad)
            angle_source = f"navigation NetPitch (Δt: {min_time_diff.total_seconds():.3f}s)"

    # Direct file load if still no angle
    if abs(net_angle_deg) < 0.1:
        nav_file = Path(exports_folder) / "by_bag" / f"navigation_plane_approximation__{target_bag}_data.csv"
        if nav_file.exists():
            direct_nav_df = pd.read_csv(nav_file)
            direct_nav_df['timestamp'] = pd.to_datetime(direct_nav_df['ts_utc'])

            direct_time_diffs = abs(direct_nav_df['timestamp'] - sonar_timestamp)
            direct_min_diff = direct_time_diffs.min()

            if direct_min_diff <= pd.Timedelta('5s'):
                direct_idx = direct_time_diffs.idxmin()
                # Invert sign of NetPitch coming from direct file as well
                net_angle_rad = -direct_nav_df.loc[direct_idx, 'NetPitch']
                net_angle_deg = np.degrees(net_angle_rad)
                

    # Create enhanced visualization - HORIZONTAL LINE ROTATED BY NET PITCH (LOCAL)
    fig, ax = plt.subplots(figsize=(18, 14))

    # Display the sonar cone
    im = ax.imshow(processed_cone, extent=extent, origin='lower', cmap='viridis', alpha=0.8)

    # Focus ONLY on Navigation NetDistance with proper rotation
    if distance_data and 'Navigation NetDistance' in distance_data:
        distance = distance_data['Navigation NetDistance']['value']

        if distance <= sonar_params['rmax']:
            # Create a net line where 0 = parallel to x-axis (cross-track)
            net_half_width = 2.0  # Half width of the net line

            # Original cross-track oriented line points (0 reference = parallel to x-axis)
            original_x1 = -net_half_width
            original_y1 = 0  # Start at origin level
            original_x2 = net_half_width
            original_y2 = 0  # Start at origin level

            # Rotate by net pitch angle directly (0 = parallel to x-axis)
            cos_angle = np.cos(net_angle_rad)   # Use direct angle (no negation)
            sin_angle = np.sin(net_angle_rad)   # Use direct angle (no negation)

            # Apply rotation matrix to both endpoints
            rotated_x1 = original_x1 * cos_angle - original_y1 * sin_angle
            rotated_y1 = original_x1 * sin_angle + original_y1 * cos_angle
            rotated_x2 = original_x2 * cos_angle - original_y2 * sin_angle
            rotated_y2 = original_x2 * sin_angle + original_y2 * cos_angle

            # Translate the rotated line to the net distance
            rotated_x1 += 0  # No x offset
            rotated_y1 += distance  # Move to net distance
            rotated_x2 += 0  # No x offset
            rotated_y2 += distance  # Move to net distance

            # Draw ONLY the rotated net line (thinner orange line)
            ax.plot([rotated_x1, rotated_x2], [rotated_y1, rotated_y2],
                   color='orange', linewidth=4, alpha=0.95,
                   label=f" Net Position: {distance:.2f}m @ {net_angle_deg:.1f} (pitch)", zorder=5)

            # Add smaller endpoints
            ax.plot([rotated_x1, rotated_x2], [rotated_y1, rotated_y2], 'o',
                   color='orange', markersize=8, markeredgecolor='white',
                   markeredgewidth=2, zorder=6)

    # Simplified range rings (no white text boxes)
    range_rings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    for r in range_rings:
        if r <= extent[3]:
            circle = patches.Circle((0, 0), r, fill=False, color='cyan',
                                  alpha=0.4, linewidth=1, linestyle='--', zorder=1)
            ax.add_patch(circle)

    # Simplified bearing lines (thinner)
    angles = np.arange(-75, 76, 15)
    for angle in angles:
        if angle != 0:
            angle_rad = np.radians(angle)
            x_end_bearing = extent[3] * np.sin(angle_rad)
            y_end_bearing = extent[3] * np.cos(angle_rad)
            ax.plot([0, x_end_bearing], [0, y_end_bearing], color='cyan', alpha=0.3,
                   linewidth=0.8, linestyle=':', zorder=1)

    # ENHANCED FORMATTING
    ax.set_xlabel('Cross-track Distance (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Forward Distance (m)', fontsize=16, fontweight='bold')

    # ENHANCED TITLE
    title = f" SONAR WITH NET POSITION (PITCH-BASED) - {target_bag}\n"
    title += f"Frame {frame_index} | rmax={sonar_params['rmax']}m"
    if distance_data and 'Navigation NetDistance' in distance_data:
        title += f" | Net: {distance_data['Navigation NetDistance']['value']:.2f}m @ {net_angle_deg:.1f} (pitch)"
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25)

    # SIMPLIFIED LEGEND
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95,
             fancybox=True, shadow=False, borderpad=0.8,
             facecolor='white', edgecolor='gray')

    # Colorbar (no grid)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Sonar Intensity', fontsize=14, fontweight='bold')

    # Set aspect and limits
    ax.set_aspect('equal')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # SIMPLIFIED INFO BOX
    if distance_data and 'Navigation NetDistance' in distance_data:
        info_text = f"🎯 NET POSITION:\n"
        info_text += f"• Distance: {distance_data['Navigation NetDistance']['value']:.2f}m\n"
        info_text += f"• Pitch: {net_angle_deg:.1f} (direct)\n"
        info_text += f"• Orange line = net position"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                alpha=0.9, edgecolor='gray', linewidth=1), zorder=10)

    plt.tight_layout()
    plt.show()
