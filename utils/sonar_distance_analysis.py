# 🎯 SONAR DISTANCE ANALYSIS UTILITIES
# ===================================
# 
# Utility functions for synchronized sonar and distance measurement analysis
# 
# Key Functions:
# - load_all_distance_data_for_bag(): Load all sensor data for a specific bag
# - create_sonar_visualization(): Generate sonar visualization with distance overlays
# - analyze_distance_measurements(): Statistical analysis of distance data
# - search_csv_files_for_measurements(): Search for measurement columns across CSV files

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple, Optional, List
import warnings

# Import sonar utilities
import utils.sonar_utils as sonar_utils

def load_all_distance_data_for_bag(target_bag: str, exports_folder: str = "/Users/eirikvarnes/code/SOLAQUA/exports") -> Tuple[Dict, Dict]:
    """
    Load all distance measurement data for a specific bag.
    
    Args:
        target_bag: Bag identifier (e.g., "2024-08-22_14-29-05")
        exports_folder: Path to exports folder
        
    Returns:
        Tuple of (raw_data_dict, distance_measurements_dict)
    """
    
    print(f"🎯 LOADING ALL DISTANCE DATA FOR BAG: {target_bag}")
    print("=" * 60)
    
    data_folder = Path(exports_folder)
    distance_measurements = {}
    raw_data = {}
    
    # 1. Load Navigation Data
    print("📡 1. Loading Navigation Data...")
    try:
        nav_file = data_folder / "by_bag" / f"navigation_plane_approximation__{target_bag}_data.csv"
        if nav_file.exists():
            nav_data = pd.read_csv(nav_file, usecols=['ts_oslo', 'NetDistance', 'Altitude'])
            nav_data['timestamp'] = pd.to_datetime(nav_data['ts_oslo']).dt.tz_convert('UTC')
            nav_data['net_distance_m_raw'] = nav_data['NetDistance']
            raw_data['navigation'] = nav_data[['timestamp', 'net_distance_m_raw', 'NetDistance', 'Altitude']].copy()
            print(f"   ✅ Loaded {len(nav_data)} navigation records")
        else:
            print(f"   ❌ Navigation file not found")
            raw_data['navigation'] = None
    except Exception as e:
        print(f"   ❌ Error loading navigation: {e}")
        raw_data['navigation'] = None
    
    # 2. Load Guidance Data  
    print("📡 2. Loading Guidance Data...")
    try:
        guidance_file = data_folder / "by_bag" / f"guidance__{target_bag}_data.csv"
        if guidance_file.exists():
            guidance_data = pd.read_csv(guidance_file)
            guidance_data['timestamp'] = pd.to_datetime(guidance_data['ts_oslo']).dt.tz_convert('UTC')
            distance_cols = [col for col in guidance_data.columns if 'distance' in col.lower()]
            if distance_cols:
                raw_data['guidance'] = guidance_data[['timestamp'] + distance_cols].copy()
                print(f"   ✅ Loaded {len(guidance_data)} guidance records with {distance_cols}")
            else:
                raw_data['guidance'] = None
                print(f"   ⚠️  No distance columns in guidance data")
        else:
            print(f"   ❌ Guidance file not found")
            raw_data['guidance'] = None
    except Exception as e:
        print(f"   ❌ Error loading guidance: {e}")
        raw_data['guidance'] = None
    
    # 3. Load DVL Altimeter
    print("📡 3. Loading DVL Altimeter...")
    try:
        dvl_alt_file = data_folder / "by_bag" / f"nucleus1000dvl_altimeter__{target_bag}_data.csv"
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
                print(f"   ✅ Loaded {len(dvl_alt)} DVL altimeter records")
            else:
                print(f"   ⚠️  No altimeter_distance column")
        else:
            print(f"   ❌ DVL altimeter file not found")
    except Exception as e:
        print(f"   ❌ Error loading DVL altimeter: {e}")
    
    # 4. Load USBL
    print("📡 4. Loading USBL...")
    try:
        usbl_file = data_folder / "by_bag" / f"sensor_usbl__{target_bag}_data.csv"
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
                print(f"   ✅ Loaded {len(usbl)} USBL records")
            else:
                print(f"   ⚠️  Missing USBL position columns")
        else:
            print(f"   ❌ USBL file not found")
    except Exception as e:
        print(f"   ❌ Error loading USBL: {e}")
    
    # 5. Load DVL Position
    print("📡 5. Loading DVL Position...")
    try:
        dvl_pos_file = data_folder / "by_bag" / f"sensor_dvl_position__{target_bag}_data.csv"
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
                print(f"   ✅ Loaded {len(dvl_pos)} DVL position records")
            else:
                print(f"   ⚠️  Missing DVL position columns")
        else:
            print(f"   ❌ DVL position file not found")
    except Exception as e:
        print(f"   ❌ Error loading DVL position: {e}")
    
    # 6. Load Navigation Position
    print("📡 6. Loading Navigation Position...")
    try:
        nav_pos_file = data_folder / "by_bag" / f"navigation_plane_approximation_position__{target_bag}_data.csv"
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
                print(f"   ✅ Loaded {len(nav_pos)} navigation position records")
            else:
                print(f"   ⚠️  Missing navigation position columns")
        else:
            print(f"   ❌ Navigation position file not found")
    except Exception as e:
        print(f"   ❌ Error loading navigation position: {e}")
    
    # 7. Load INS Z Position
    print("📡 7. Loading INS Z Position...")
    try:
        ins_file = data_folder / "by_bag" / f"nucleus1000dvl_ins__{target_bag}_data.csv"
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
                    print(f"   ✅ Loaded {len(ins_data)} INS records with {z_position_col}")
                else:
                    print(f"   ⚠️  No valid {z_position_col} values")
            else:
                print(f"   ⚠️  No Z position columns found")
        else:
            print(f"   ❌ INS file not found")
    except Exception as e:
        print(f"   ❌ Error loading INS: {e}")
    
    # Summary
    print(f"\n📊 LOADING SUMMARY:")
    print(f"   🎯 Target bag: {target_bag}")
    print(f"   📁 Raw data loaded: {len([k for k, v in raw_data.items() if v is not None])}/{len(raw_data)}")
    print(f"   📏 Distance measurements: {len(distance_measurements)}")
    
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


def create_sonar_visualization(target_bag, frame_idx, raw_nav_data, raw_guidance_data, distance_measurements, 
                             exports_folder="/Users/eirikvarnes/code/SOLAQUA/exports", figsize=(16, 12)):
    """
    Create sonar visualization with distance measurement overlays.
    
    Args:
        target_bag: Bag identifier
        frame_idx: Frame index to visualize
        raw_nav_data: Navigation DataFrame
        raw_guidance_data: Guidance DataFrame
        distance_measurements: Dictionary of sensor measurements
        exports_folder: Path to exports folder
        figsize: Figure size tuple
        
    Returns:
        matplotlib figure or None if error
    """
    
    print(f"🎯 CREATING SONAR VISUALIZATION FOR BAG: {target_bag}")
    print(f"📊 Frame: {frame_idx}")
    print("=" * 50)
    
    try:
        import utils.sonar_utils as sonar_utils
        from utils.sonar_utils import ConeGridSpec
        
        # Build sonar file path for current bag
        sonar_csv_file = Path(exports_folder) / "by_bag" / f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
        
        if not sonar_csv_file.exists():
            print(f"❌ Sonar CSV file not found for bag {target_bag}")
            print(f"   Expected: {sonar_csv_file}")
            return None
        
        # Load sonar data for current bag
        sonar_df = pd.read_csv(sonar_csv_file)
        print(f"📡 Loaded {len(sonar_df)} sonar frames for bag {target_bag}")
        
        if frame_idx >= len(sonar_df):
            print(f"❌ Frame {frame_idx} not available (only {len(sonar_df)} frames)")
            print(f"   Adjusting to last available frame: {len(sonar_df)-1}")
            frame_idx = len(sonar_df) - 1
        
        # Configure 5m sonar processing
        sonar_params = {
            'fov_deg': 120,
            'rmin': 0.0,
            'rmax': 20.0,
            'y_zoom': 5.0,  # 5m range!
            'grid': ConeGridSpec(img_w=900, img_h=700),
            'enhanced': True
        }
        
        # Get the raw sonar frame
        M = sonar_utils.get_sonoptix_frame(sonar_df, frame_idx)
        if M is None:
            print(f"❌ Could not get sonar frame {frame_idx}")
            return None
        
        # Enhance and process
        Z = sonar_utils.enhance_intensity(M, sonar_params['rmin'], sonar_params['rmax'])
        cone, extent = sonar_utils.rasterize_cone(
            Z, fov_deg=sonar_params['fov_deg'], 
            rmin=sonar_params['rmin'], 
            rmax=sonar_params['rmax'], 
            y_zoom=sonar_params['y_zoom'], 
            grid=sonar_params['grid']
        )
        
        # Get timestamp
        sonar_timestamp = pd.to_datetime(sonar_df.loc[frame_idx, 'ts_utc'])
        print(f"   🕐 Time: {sonar_timestamp.strftime('%H:%M:%S')}")
        print(f"   📐 Sonar extent: {extent}")
        
        # Collect all available distance measurements at this timestamp
        distance_data = collect_distance_measurements_at_timestamp(
            sonar_timestamp, raw_nav_data, raw_guidance_data, distance_measurements
        )
        
        print(f"\n📏 Found {len(distance_data)} distance measurements:")
        for name, data in distance_data.items():
            emoji = "🧭" if "INS" in name else "📏"
            print(f"   {emoji} {name}: {data['value']:.3f}m ({data['description']})")
        
        # Create visualization
        fig = plt.figure(figsize=figsize)
        
        # Show the sonar image
        im = plt.imshow(cone, extent=extent, cmap='viridis', origin='lower')
        
        # Add all distance lines with slight offsets to prevent overlap
        y_offset = 0.02  # Small vertical offset
        current_offset = 0
        
        for i, (name, data) in enumerate(distance_data.items()):
            # Special highlighting for INS data
            if "INS" in name:
                line_alpha = 0.9
                line_width = data['width'] + 0.5
            else:
                line_alpha = 0.85
                line_width = data['width']
            
            # Alternate offsets to spread lines
            if i > 0:
                current_offset = y_offset * (i % 3 - 1)
            
            plt.axhline(
                y=data['value'] + current_offset, 
                color=data['color'], 
                linewidth=line_width,
                linestyle=data['style'],
                label=f"{'🧭 ' if 'INS' in name else ''}{name}: {data['value']:.3f}m",
                alpha=line_alpha
            )
        
        # Add range rings for reference
        range_rings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        for r in range_rings:
            if r <= extent[3]:
                circle = plt.Circle((0, 0), r, fill=False, color='cyan', alpha=0.3, linewidth=0.8, linestyle='--')
                plt.gca().add_patch(circle)
                plt.text(0.1, r-0.05, f'{r}m', color='cyan', fontsize=9, alpha=0.7)
        
        # Add polar grid lines for bearing reference
        angles = np.arange(-60, 61, 15)
        for angle in angles:
            if angle != 0:
                angle_rad = np.radians(angle)
                x_end = extent[3] * np.sin(angle_rad)
                y_end = extent[3] * np.cos(angle_rad)
                plt.plot([0, x_end], [0, y_end], color='cyan', alpha=0.2, linewidth=0.5, linestyle=':')
                if abs(angle) == 30 or abs(angle) == 60:
                    plt.text(x_end*0.9, y_end*0.9, f'{angle}°', color='cyan', fontsize=8, alpha=0.7, 
                            ha='center', va='center')
        
        # Formatting
        plt.xlabel('Starboard Distance (m)', fontsize=12)
        plt.ylabel('Forward Distance (m)', fontsize=12) 
        plt.title(f'Bag: {target_bag} | Frame {frame_idx} | Time: {sonar_timestamp.strftime("%H:%M:%S")}\n{len(distance_data)} Distance Sources', fontsize=14)
        
        # Create legend
        legend = plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Sonar Intensity', fontsize=11)
        
        # Set equal aspect ratio
        plt.axis('equal')
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        
        plt.tight_layout()
        
        print(f"\n✅ Visualization complete for bag: {target_bag}")
        return fig
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_distance_measurements(distance_data):
    """
    Perform statistical analysis on distance measurements.
    
    Args:
        distance_data: Dictionary of distance measurements
        
    Returns:
        Dictionary with analysis results
    """
    
    if not distance_data:
        return {"error": "No distance data provided"}
    
    analysis = {}
    
    # Basic statistics
    distances = [data['value'] for data in distance_data.values()]
    distance_names = list(distance_data.keys())
    
    analysis['basic_stats'] = {
        'count': len(distances),
        'min': min(distances),
        'max': max(distances),
        'range': max(distances) - min(distances),
        'mean': np.mean(distances),
        'std': np.std(distances)
    }
    
    # Special analysis for INS measurements
    ins_measurements = {name: data for name, data in distance_data.items() if 'INS' in name}
    analysis['ins_analysis'] = ins_measurements
    
    # Look for potential offsets/biases
    if 'Navigation NetDistance' in distance_data:
        nav_dist = distance_data['Navigation NetDistance']['value']
        analysis['offset_analysis'] = {}
        analysis['offset_analysis']['reference'] = nav_dist
        analysis['offset_analysis']['offsets'] = {}
        
        for name, data in distance_data.items():
            if name != 'Navigation NetDistance':
                offset = data['value'] - nav_dist
                analysis['offset_analysis']['offsets'][name] = offset
    
    # Identify measurements close to each other (clustering)
    close_pairs = []
    for i, (name1, data1) in enumerate(distance_data.items()):
        for j, (name2, data2) in enumerate(distance_data.items()):
            if i < j:
                diff = abs(data1['value'] - data2['value'])
                if diff <= 0.1:  # Within 10cm
                    close_pairs.append((name1, name2, diff))
    
    analysis['clustering'] = close_pairs
    
    return analysis


def search_csv_files_for_measurements(target_bag, exports_folder="/Users/eirikvarnes/code/SOLAQUA/exports"):
    """
    Search CSV files for measurement-related columns.
    
    Args:
        target_bag: Bag identifier
        exports_folder: Path to exports folder
        
    Returns:
        Dictionary with search findings
    """
    
    by_bag_folder = Path(exports_folder) / "by_bag"
    
    # Search keywords for different types of measurements
    search_keywords = {
        'net': ['net', 'Net', 'NET'],
        'distance': ['distance', 'Distance', 'DISTANCE'],
        'depth': ['depth', 'Depth', 'DEPTH'],
        'range': ['range', 'Range', 'RANGE'],
        'position': ['position', 'Position', 'x', 'y', 'z', 'east', 'north'],
        'altitude': ['altitude', 'Altitude', 'alt', 'Alt']
    }
    
    findings = {category: {} for category in search_keywords.keys()}
    
    # Get all CSV files for the target bag
    csv_files = list(by_bag_folder.glob(f"*{target_bag}*.csv"))
    
    print(f"🔍 Searching {len(csv_files)} CSV files for bag {target_bag}...")
    
    for csv_file in csv_files:
        try:
            # Read just the first row to get column names
            df_sample = pd.read_csv(csv_file, nrows=1)
            columns = df_sample.columns.tolist()
            
            file_name = csv_file.stem
            
            # Search for keywords in column names
            for category, keywords in search_keywords.items():
                matching_cols = []
                for col in columns:
                    for keyword in keywords:
                        if keyword in col:
                            matching_cols.append(col)
                
                if matching_cols:
                    findings[category][file_name] = matching_cols
                    
        except Exception as e:
            print(f"   ⚠️  Error reading {csv_file.name}: {e}")
    
    # Summary
    total_files_with_findings = len(set().union(*[files.keys() for files in findings.values()]))
    
    print(f"\n📊 Search Summary:")
    print(f"   📁 Files searched: {len(csv_files)}")
    print(f"   📋 Files with findings: {total_files_with_findings}")
    
    for category, files in findings.items():
        if files:
            print(f"   📏 {category.title()} measurements: {len(files)} files")
    
    return findings


# Convenience function for quick analysis
def quick_sonar_analysis(target_bag, frame_idx=500, exports_folder="/Users/eirikvarnes/code/SOLAQUA/exports"):
    """
    Quick all-in-one sonar analysis for a bag.
    
    Args:
        target_bag: Bag identifier
        frame_idx: Frame to analyze
        exports_folder: Path to exports folder
        
    Returns:
        Tuple of (figure, analysis_results)
    """
    
    print(f"🚀 QUICK SONAR ANALYSIS FOR BAG: {target_bag}")
    print("=" * 50)
    
    # Load all data
    raw_data, distance_measurements = load_all_distance_data_for_bag(target_bag, exports_folder)
    
    # Create visualization
    fig = create_sonar_visualization(
        target_bag, frame_idx, 
        raw_data['navigation'], 
        raw_data['guidance'], 
        distance_measurements, 
        exports_folder
    )
    
    if fig:
        plt.show()
    
    return fig, {"raw_data": raw_data, "distance_measurements": distance_measurements}


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


def analyze_sonar_data_range(target_bag: str, exports_folder: str = "/Users/eirikvarnes/code/SOLAQUA/exports") -> Dict:
    """
    Analyze the raw sonar data to help determine appropriate rmax settings.
    This will help you manually determine what range was actually used.
    
    Args:
        target_bag: Bag identifier
        exports_folder: Path to exports folder
        
    Returns:
        Dictionary with analysis results to help determine rmax
    """
    
    import utils.sonar_utils as sonar_utils
    
    print(f"🔍 ANALYZING SONAR DATA RANGE FOR BAG: {target_bag}")
    print("=" * 50)
    
    # Load sonar data
    sonar_csv_file = Path(exports_folder) / "by_bag" / f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
    
    if not sonar_csv_file.exists():
        return {"error": f"Sonar data file not found: {sonar_csv_file}"}
    
    try:
        print(f"📡 Loading sonar data from: {sonar_csv_file.name}")
        sonar_df = pd.read_csv(sonar_csv_file)
        print(f"   ✅ Loaded {len(sonar_df)} sonar frames")
    except Exception as e:
        return {"error": f"Error loading sonar data: {e}"}
    
    # Analyze a sample of frames
    sample_size = min(50, len(sonar_df))
    sample_indices = np.linspace(0, len(sonar_df)-1, sample_size, dtype=int)
    
    print(f"🔬 Analyzing {sample_size} sample frames...")
    
    matrix_shapes = []
    matrix_ranges = []
    max_beam_counts = []
    max_range_counts = []
    
    for i, idx in enumerate(sample_indices):
        try:
            M = sonar_utils.get_sonoptix_frame(sonar_df, idx)
            if M is not None:
                matrix_shapes.append(M.shape)
                matrix_ranges.append((M.min(), M.max()))
                
                # Assume rows = range bins, cols = beams
                max_range_counts.append(M.shape[0])  # Number of range bins
                max_beam_counts.append(M.shape[1])   # Number of beams
                
        except Exception as e:
            print(f"   ⚠️  Error processing frame {idx}: {e}")
    
    if not matrix_shapes:
        return {"error": "Could not extract any sonar frames"}
    
    # Analysis results
    analysis = {
        'bag': target_bag,
        'total_frames': len(sonar_df),
        'analyzed_frames': len(matrix_shapes),
        'matrix_shapes': {
            'most_common': max(set(matrix_shapes), key=matrix_shapes.count),
            'all_shapes': list(set(matrix_shapes))
        },
        'range_bins': {
            'min': min(max_range_counts),
            'max': max(max_range_counts),
            'most_common': max(set(max_range_counts), key=max_range_counts.count)
        },
        'beam_count': {
            'min': min(max_beam_counts),
            'max': max(max_beam_counts),
            'most_common': max(set(max_beam_counts), key=max_beam_counts.count)
        },
        'intensity_range': {
            'min_value': min([r[0] for r in matrix_ranges]),
            'max_value': max([r[1] for r in matrix_ranges]),
            'typical_range': (
                np.mean([r[0] for r in matrix_ranges]),
                np.mean([r[1] for r in matrix_ranges])
            )
        }
    }
    
    # Suggest possible rmax values based on range bins
    common_range_bins = analysis['range_bins']['most_common']
    
    # Common sonar range configurations
    possible_ranges = []
    if common_range_bins >= 1000:
        possible_ranges.extend([30.0, 50.0, 100.0])
    elif common_range_bins >= 500:
        possible_ranges.extend([20.0, 30.0])
    elif common_range_bins >= 200:
        possible_ranges.extend([10.0, 15.0, 20.0])
    else:
        possible_ranges.extend([5.0, 10.0])
    
    analysis['suggested_rmax'] = possible_ranges
    
    # Print analysis
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   🖼️  Most common matrix shape: {analysis['matrix_shapes']['most_common']}")
    print(f"   📏 Range bins (rows): {analysis['range_bins']['most_common']}")
    print(f"   📡 Beam count (cols): {analysis['beam_count']['most_common']}")
    print(f"   📊 Intensity range: {analysis['intensity_range']['typical_range'][0]:.1f} to {analysis['intensity_range']['typical_range'][1]:.1f}")
    
    print(f"\n💡 SUGGESTED RMAX VALUES:")
    print(f"   Based on {analysis['range_bins']['most_common']} range bins:")
    for rmax in analysis['suggested_rmax']:
        range_resolution = rmax / analysis['range_bins']['most_common']
        print(f"   • {rmax}m (resolution: {range_resolution:.3f}m per bin)")
    
    print(f"\n🎯 TO DETERMINE ACTUAL RMAX:")
    print(f"   1. Check your experimental notes/configuration")
    print(f"   2. Look at sonar hardware documentation") 
    print(f"   3. Try different rmax values and see which gives sensible distance scaling")
    print(f"   4. Compare with known distance measurements in your data")
    
    return analysis


def extract_raw_sonar_data_with_configurable_rmax(
    target_bag: str,
    frame_idx: int,
    rmax: float = None,
    exports_folder: str = "/Users/eirikvarnes/code/SOLAQUA/exports"
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
    from utils.sonar_utils import ConeGridSpec
    
    print(f"🔬 EXTRACTING RAW SONAR DATA WITH CONFIGURABLE RMAX")
    print(f"   📁 Bag: {target_bag}")
    print(f"   🖼️  Frame: {frame_idx}")
    print(f"   📏 rmax: {rmax if rmax else 'default (20.0m)'}")
    print("=" * 50)
    
    # 1. Get configurable sonar parameters
    sonar_params = get_configurable_sonar_parameters(target_bag, rmax)
    
    print(f"📊 SONAR PARAMETERS:")
    print(f"   🏷️  Run Type: {sonar_params['run_type']}")
    print(f"   📏 Range Max: {sonar_params['rmax']}m ({sonar_params['range_source']})")
    print(f"   📝 Description: {sonar_params['description']}")
    
    # 2. Load raw sonar data
    sonar_csv_file = Path(exports_folder) / "by_bag" / f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
    
    if not sonar_csv_file.exists():
        print(f"❌ Sonar data file not found: {sonar_csv_file}")
        return None, None, {}, sonar_params
    
    try:
        print(f"📡 Loading sonar data from: {sonar_csv_file.name}")
        sonar_df = pd.read_csv(sonar_csv_file)
        print(f"   ✅ Loaded {len(sonar_df)} sonar frames")
    except Exception as e:
        print(f"❌ Error loading sonar data: {e}")
        return None, None, {}, sonar_params
    
    # 3. Validate frame index
    if frame_idx >= len(sonar_df):
        print(f"❌ Frame {frame_idx} not available (only {len(sonar_df)} frames)")
        frame_idx = min(frame_idx, len(sonar_df) - 1)
        print(f"   🔄 Adjusted to frame: {frame_idx}")
    
    # 4. Extract raw sonar frame matrix
    try:
        print(f"🔍 Extracting raw sonar frame {frame_idx}...")
        raw_matrix = sonar_utils.get_sonoptix_frame(sonar_df, frame_idx)
        
        if raw_matrix is None:
            print(f"❌ Could not extract sonar frame {frame_idx}")
            return None, None, {}, sonar_params
            
        print(f"   ✅ Raw matrix shape: {raw_matrix.shape}")
        print(f"   📊 Value range: {raw_matrix.min():.3f} to {raw_matrix.max():.3f}")
        
    except Exception as e:
        print(f"❌ Error extracting raw sonar frame: {e}")
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
        
        print(f"   ✅ Processed cone shape: {cone.shape}")
        print(f"   📐 Extent: {extent}")
        
        # Get timestamp
        sonar_timestamp = pd.to_datetime(sonar_df.loc[frame_idx, 'ts_utc'])
        print(f"   🕐 Timestamp: {sonar_timestamp.strftime('%H:%M:%S')}")
        
        return raw_matrix, cone, extent, sonar_params
        
    except Exception as e:
        print(f"❌ Error processing sonar data: {e}")
        return raw_matrix, None, {}, sonar_params


def create_enhanced_sonar_visualization_with_configurable_rmax(
    target_bag: str,
    frame_idx: int,
    rmax: float = None,
    raw_nav_data: pd.DataFrame = None,
    raw_guidance_data: pd.DataFrame = None, 
    distance_measurements: Dict = None,
    exports_folder: str = "/Users/eirikvarnes/code/SOLAQUA/exports",
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Create enhanced sonar visualization using raw data extraction with configurable rmax settings.
    
    Args:
        target_bag: Bag identifier
        frame_idx: Frame index to visualize
        rmax: Maximum range in meters (if None, defaults to 20.0m)
        raw_nav_data: Navigation data (optional, will load if None)
        raw_guidance_data: Guidance data (optional, will load if None) 
        distance_measurements: Distance measurement data (optional, will load if None)
        exports_folder: Path to exports folder
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object or None if failed
    """
    
    print(f"🎨 CREATING ENHANCED SONAR VISUALIZATION WITH CONFIGURABLE RMAX")
    print(f"   📏 rmax: {rmax if rmax else 'default (20.0m)'}")
    print("=" * 60)
    
    # Extract raw sonar data with configurable rmax
    raw_matrix, cone, extent, sonar_params = extract_raw_sonar_data_with_configurable_rmax(
        target_bag, frame_idx, rmax, exports_folder
    )
    
    if cone is None:
        print(f"❌ Failed to extract sonar data")
        return None
    
    # Load distance data if not provided
    if raw_nav_data is None or distance_measurements is None:
        print(f"📡 Loading distance measurement data...")
        raw_data, distance_measurements = load_all_distance_data_for_bag(target_bag, exports_folder)
        raw_nav_data = raw_data.get('navigation')
        raw_guidance_data = raw_data.get('guidance')
    
    # Get sonar timestamp for synchronization
    sonar_csv_file = Path(exports_folder) / "by_bag" / f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
    sonar_df = pd.read_csv(sonar_csv_file)
    sonar_timestamp = pd.to_datetime(sonar_df.loc[frame_idx, 'ts_utc'])
    
    # Collect synchronized distance measurements
    distance_data = collect_distance_measurements_at_timestamp(
        sonar_timestamp, raw_nav_data, raw_guidance_data, distance_measurements
    )
    
    print(f"📊 Found {len(distance_data)} synchronized measurements")
    
    # Create the visualization using the processed cone
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the sonar cone
    x_min, x_max, y_min, y_max = extent
    im = ax.imshow(cone, extent=extent, origin='lower', cmap='viridis', alpha=0.8)
    plt.colorbar(im, ax=ax, label='Sonar Intensity')
    
    # Add title with run information
    range_info = f"rmax={sonar_params['rmax']}m ({sonar_params['range_source']})"
    title = f"🎯 Enhanced Sonar Visualization - {target_bag}\n"
    title += f"Frame {frame_idx} | {sonar_params['run_type']} | {range_info}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add distance measurements as overlays
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    legend_elements = []
    
    for i, (name, data) in enumerate(distance_data.items()):
        color = colors[i % len(colors)]
        distance = data['value']
        
        # Draw distance line (straight ahead)
        if distance <= sonar_params['rmax']:  # Only show if within range
            ax.plot([0, 0], [0, distance], color=color, linewidth=3, alpha=0.8)
            ax.plot(0, distance, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=f"{name}: {distance:.2f}m"))
    
    # Add range rings
    range_rings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5] 
    for r in range_rings:
        if r <= sonar_params['y_zoom']:
            circle = patches.Circle((0, 0), r, fill=False, color='cyan', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
            ax.text(0.1, r-0.1, f"{r}m", color='cyan', fontsize=8, alpha=0.7)
    
    # Add angular guidelines
    angles = np.arange(-60, 61, 15)
    for angle in angles:
        rad = np.radians(angle)
        x_end = sonar_params['y_zoom'] * np.sin(rad)
        y_end = sonar_params['y_zoom'] * np.cos(rad)
        ax.plot([0, x_end], [0, y_end], 'white', alpha=0.2, linewidth=0.5)
    
    # Set equal aspect and limits
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Cross-track Distance (m)')
    ax.set_ylabel('Forward Distance (m)')
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add parameter info text box
    info_text = f"Run Parameters:\n"
    info_text += f"• Type: {sonar_params['run_type']}\n"
    info_text += f"• Range: {sonar_params['rmin']}-{sonar_params['rmax']}m ({sonar_params['range_source']})\n"
    info_text += f"• FOV: {sonar_params['fov_deg']}°\n"
    info_text += f"• Zoom: {sonar_params['y_zoom']}m\n"
    info_text += f"• Raw matrix: {raw_matrix.shape if raw_matrix is not None else 'N/A'}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    print(f"✅ Enhanced visualization complete with {sonar_params['range_source']} rmax={sonar_params['rmax']}m!")
    
    return fig


def create_enhanced_sonar_plot_with_measurements(
    cone: np.ndarray,
    extent: Tuple[float, float, float, float],
    distance_data: Dict,
    sonar_params: Dict,
    frame_idx: int = 0,
    timestamp: str = None,
    figsize: Tuple[int, int] = (16, 12),
    show_range_rings: bool = True,
    show_bearing_lines: bool = True,
    title_prefix: str = "Enhanced Sonar Visualization",
    net_angle_rad: float = 0.0,
    angle_source: str = "default"
):
    """
    Create enhanced sonar visualization with distance measurement overlays.
    
    Args:
        cone: Processed sonar cone data
        extent: Spatial extent [x_min, x_max, y_min, y_max]
        distance_data: Dictionary of distance measurements
        sonar_params: Sonar processing parameters
        frame_idx: Frame index for title
        timestamp: Timestamp string for title
        figsize: Figure size tuple
        show_range_rings: Whether to show range rings
        show_bearing_lines: Whether to show bearing lines
        title_prefix: Prefix for plot title
        net_angle_rad: Net heading angle in radians (0.0 = straight ahead)
        angle_source: Description of angle data source
        
    Returns:
        matplotlib Figure object
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the sonar cone
    im = ax.imshow(cone, extent=extent, cmap='viridis', origin='lower', alpha=0.8)
    
    # Convert angle to degrees for display
    net_angle_deg = np.degrees(net_angle_rad)
    
    # Add distance measurement lines
    if distance_data:
        y_offset = 0.02  # Small vertical offset to prevent overlap
        current_offset = 0
        legend_elements = []
        
        for i, (name, data) in enumerate(distance_data.items()):
            color = data.get('color', 'red')
            style = data.get('style', '-')
            width = data.get('width', 2)
            distance = data['value']
            
            # Special handling for different measurement types
            if "INS" in name:
                line_alpha = 0.9
                line_width = width + 0.5
                marker_size = 10
                emoji = "🧭"
            elif "Navigation" in name:
                line_alpha = 0.9
                line_width = width
                marker_size = 8
                emoji = "🎯"
            elif "Guidance" in name or "Desired" in name:
                line_alpha = 0.8
                line_width = width
                marker_size = 7
                emoji = "🎮"
            else:
                line_alpha = 0.7
                line_width = width
                marker_size = 6
                emoji = "📏"
            
            # Alternate offsets to spread lines
            if i > 0:
                current_offset = y_offset * (i % 3 - 1)
            
            # Only draw if within sonar range
            if distance <= sonar_params['rmax']:
                
                # For Navigation NetDistance, use the actual net angle
                if 'Navigation' in name and 'NetDistance' in name:
                    # Calculate end point using net angle
                    x_end = distance * np.sin(net_angle_rad)
                    y_end = distance * np.cos(net_angle_rad)
                    
                    # Draw angled line to net
                    ax.plot([0, x_end], [0, y_end], color=color, linewidth=line_width+1, 
                           linestyle=style, alpha=line_alpha)
                    ax.plot(x_end, y_end, 'o', color=color, markersize=marker_size+2, 
                           markeredgecolor='white', markeredgewidth=2)
                    
                    # Add angle arc indicator if angle is significant
                    if abs(net_angle_deg) > 2:
                        arc_angles = np.linspace(0, net_angle_rad, 20)
                        arc_radius = distance * 0.15  # 15% of distance
                        arc_x = arc_radius * np.sin(arc_angles)
                        arc_y = arc_radius * np.cos(arc_angles)
                        ax.plot(arc_x, arc_y, color=color, linestyle='--', alpha=0.6, linewidth=2)
                    
                    # Special label for angled measurement
                    label = f"{emoji} {name}: {distance:.2f}m @ {net_angle_deg:.1f}°"
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=line_width+1, 
                                                    linestyle=style, label=label))
                else:
                    # Other measurements - draw horizontal line (forward distance)
                    ax.axhline(
                        y=distance + current_offset,
                        color=color,
                        linewidth=line_width,
                        linestyle=style,
                        alpha=line_alpha
                    )
                    
                    # Add point marker
                    ax.plot(0, distance + current_offset, 'o', 
                           color=color, markersize=marker_size, 
                           markeredgecolor='white', markeredgewidth=1)
                    
                    # Add to legend
                    label = f"{emoji} {name}: {distance:.2f}m"
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=line_width, 
                                                    linestyle=style, label=label))
    
    # Add range rings
    if show_range_rings:
        range_rings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        for r in range_rings:
            if r <= extent[3]:  # Don't exceed sonar range
                circle = patches.Circle((0, 0), r, fill=False, color='cyan', 
                                      alpha=0.3, linewidth=0.8, linestyle='--')
                ax.add_patch(circle)
                ax.text(0.1, r-0.05, f'{r}m', color='cyan', fontsize=9, alpha=0.7)
    
    # Add bearing lines
    if show_bearing_lines:
        angles = np.arange(-60, 61, 15)  # -60 to +60 degrees every 15 degrees
        for angle in angles:
            if angle != 0:  # Skip center line
                angle_rad = np.radians(angle)
                x_end = extent[3] * np.sin(angle_rad)
                y_end = extent[3] * np.cos(angle_rad)
                ax.plot([0, x_end], [0, y_end], color='cyan', alpha=0.2, 
                       linewidth=0.5, linestyle=':')
                if abs(angle) == 30 or abs(angle) == 60:  # Label major angles
                    ax.text(x_end*0.9, y_end*0.9, f'{angle}°', color='cyan', 
                           fontsize=8, alpha=0.7, ha='center', va='center')
    
    # Formatting
    ax.set_xlabel('Cross-track Distance (m)', fontsize=12)
    ax.set_ylabel('Forward Distance (m)', fontsize=12)
    
    # Create title with angle information
    title = f"{title_prefix} - Frame {frame_idx}"
    if timestamp:
        title += f"\nTime: {timestamp}"
    if abs(net_angle_deg) > 0.1:
        title += f" | Net Heading: {net_angle_deg:.1f}° ({angle_source})"
    if distance_data:
        title += f" | {len(distance_data)} Distance Sources"
    if sonar_params:
        title += f" | rmax={sonar_params.get('rmax', 'N/A')}m"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    if 'legend_elements' in locals() and legend_elements:
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                          fontsize=10, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
    
    # Add grid and colorbar
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Sonar Intensity', fontsize=11)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # Add info text box with angle information
    if abs(net_angle_deg) > 0.1:
        info_text = f"Net Angle Info:\n"
        info_text += f"• Heading: {net_angle_deg:.1f}°\n"
        info_text += f"• Source: {angle_source}\n"
        info_text += f"• Range: {sonar_params.get('rmin', 0):.1f}-{sonar_params.get('rmax', 0):.1f}m"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig
    
    # Add range rings
    if show_range_rings:
        range_rings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        for r in range_rings:
            if r <= extent[3]:  # Don't exceed sonar range
                circle = patches.Circle((0, 0), r, fill=False, color='cyan', 
                                      alpha=0.3, linewidth=0.8, linestyle='--')
                ax.add_patch(circle)
                ax.text(0.1, r-0.05, f'{r}m', color='cyan', fontsize=9, alpha=0.7)
    
    # Add bearing lines
    if show_bearing_lines:
        angles = np.arange(-60, 61, 15)  # -60 to +60 degrees every 15 degrees
        for angle in angles:
            if angle != 0:  # Skip center line
                angle_rad = np.radians(angle)
                x_end = extent[3] * np.sin(angle_rad)
                y_end = extent[3] * np.cos(angle_rad)
                ax.plot([0, x_end], [0, y_end], color='cyan', alpha=0.2, 
                       linewidth=0.5, linestyle=':')
                if abs(angle) == 30 or abs(angle) == 60:  # Label major angles
                    ax.text(x_end*0.9, y_end*0.9, f'{angle}°', color='cyan', 
                           fontsize=8, alpha=0.7, ha='center', va='center')
    
    # Formatting
    ax.set_xlabel('Cross-track Distance (m)', fontsize=12)
    ax.set_ylabel('Forward Distance (m)', fontsize=12)
    
    # Create title
    title = f"{title_prefix} - Frame {frame_idx}"
    if timestamp:
        title += f"\nTime: {timestamp}"
    if distance_data:
        title += f" | {len(distance_data)} Distance Sources"
    if sonar_params:
        title += f" | rmax={sonar_params.get('rmax', 'N/A')}m"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    if 'legend_elements' in locals() and legend_elements:
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                          fontsize=10, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
    
    # Add grid and colorbar
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Sonar Intensity', fontsize=11)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    plt.tight_layout()
    
    return fig


def extract_net_heading_at_timestamp(timestamp, nav_data, max_time_diff=2.0):
    """
    Extract net heading angle from navigation data at a specific timestamp.
    
    Args:
        timestamp: Target timestamp (pandas Timestamp)
        nav_data: Navigation dataframe with NetHeading column
        max_time_diff: Maximum time difference in seconds to consider valid
        
    Returns:
        Tuple of (net_angle_rad, net_angle_deg, angle_source, time_offset)
    """
    if nav_data is None or 'NetHeading' not in nav_data.columns:
        return 0.0, 0.0, "default (no nav data)", 0.0
    
    # Find closest navigation measurement
    nav_time_diffs = abs(nav_data['timestamp'] - timestamp)
    min_time_diff = nav_time_diffs.min()
    
    if min_time_diff <= pd.Timedelta(f'{max_time_diff}s'):
        nav_idx = nav_time_diffs.idxmin()
        net_angle_rad = nav_data.loc[nav_idx, 'NetHeading']
        net_angle_deg = np.degrees(net_angle_rad)
        time_offset = min_time_diff.total_seconds()
        angle_source = f"navigation data (Δt: {time_offset:.2f}s)"
        return net_angle_rad, net_angle_deg, angle_source, time_offset
    else:
        return 0.0, 0.0, f"default (nav data too old: {min_time_diff.total_seconds():.1f}s)", min_time_diff.total_seconds()


def analyze_distance_measurements_clustering(
    distance_data: Dict,
    tolerance: float = 0.1,
    reference_measurement: str = None
) -> Dict:
    """
    Analyze distance measurements for clustering and offsets.
    
    Args:
        distance_data: Dictionary of distance measurements
        tolerance: Distance tolerance for clustering (meters)
        reference_measurement: Name of reference measurement for offset analysis
        
    Returns:
        Dictionary with analysis results
    """
    
    if not distance_data:
        return {'error': 'No distance data provided'}
    
    distances = [data['value'] for data in distance_data.values()]
    distance_names = list(distance_data.keys())
    
    results = {
        'total_measurements': len(distances),
        'range': {
            'min': min(distances),
            'max': max(distances),
            'spread': max(distances) - min(distances)
        },
        'close_pairs': [],
        'offsets': {},
        'statistics': {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'median': np.median(distances)
        }
    }
    
    # Find close pairs
    for i, (name1, data1) in enumerate(distance_data.items()):
        for j, (name2, data2) in enumerate(distance_data.items()):
            if i < j:
                diff = abs(data1['value'] - data2['value'])
                if diff <= tolerance:
                    results['close_pairs'].append({
                        'measurement1': name1,
                        'measurement2': name2,
                        'difference': diff,
                        'values': [data1['value'], data2['value']]
                    })
    
    # Calculate offsets relative to reference measurement
    if reference_measurement and reference_measurement in distance_data:
        ref_value = distance_data[reference_measurement]['value']
        results['reference_measurement'] = reference_measurement
        results['reference_value'] = ref_value
        
        for name, data in distance_data.items():
            if name != reference_measurement:
                offset = data['value'] - ref_value
                results['offsets'][name] = {
                    'absolute_offset': offset,
                    'relative_offset_percent': (offset / ref_value) * 100 if ref_value > 0 else 0,
                    'measurement_value': data['value']
                }
    
    # Categorize measurements by type
    results['measurement_types'] = {
        'navigation': [name for name in distance_names if 'Navigation' in name],
        'guidance': [name for name in distance_names if any(x in name for x in ['Guidance', 'Desired', 'Error'])],
        'ins': [name for name in distance_names if 'INS' in name],
        'sensors': [name for name in distance_names if not any(x in name for x in ['Navigation', 'Guidance', 'Desired', 'Error', 'INS'])]
    }
    
    return results


def print_distance_analysis_report(analysis_results: Dict, distance_data: Dict = None):
    """
    Print a comprehensive analysis report of distance measurements.
    
    Args:
        analysis_results: Results from analyze_distance_measurements_clustering
        distance_data: Original distance data for additional context
    """
    
    if 'error' in analysis_results:
        print(f"❌ {analysis_results['error']}")
        return
    
    print(f"🔍 DISTANCE MEASUREMENT ANALYSIS REPORT")
    print(f"=" * 50)
    
    # Basic statistics
    stats = analysis_results['statistics']
    range_info = analysis_results['range']
    
    print(f"📊 BASIC STATISTICS:")
    print(f"   • Total measurements: {analysis_results['total_measurements']}")
    print(f"   • Range: {range_info['min']:.3f}m to {range_info['max']:.3f}m")
    print(f"   • Spread: {range_info['spread']:.3f}m")
    print(f"   • Mean: {stats['mean']:.3f}m")
    print(f"   • Median: {stats['median']:.3f}m")
    print(f"   • Std Dev: {stats['std']:.3f}m")
    
    # Measurement types
    types = analysis_results['measurement_types']
    print(f"\n📋 MEASUREMENT CATEGORIES:")
    for category, measurements in types.items():
        if measurements:
            emoji = {"navigation": "🎯", "guidance": "🎮", "ins": "🧭", "sensors": "📏"}
            print(f"   {emoji.get(category, '📊')} {category.title()}: {', '.join(measurements)}")
    
    # Close pairs analysis
    close_pairs = analysis_results['close_pairs']
    print(f"\n🔍 CLUSTERING ANALYSIS:")
    if close_pairs:
        print(f"   📌 Measurements within tolerance:")
        for pair in close_pairs:
            print(f"     • {pair['measurement1']} ↔ {pair['measurement2']}: {pair['difference']:.3f}m difference")
            print(f"       Values: {pair['values'][0]:.3f}m, {pair['values'][1]:.3f}m")
    else:
        print(f"   📌 No measurements within {0.1:.1f}m of each other")
    
    # Offset analysis
    if 'reference_measurement' in analysis_results:
        ref_name = analysis_results['reference_measurement']
        ref_value = analysis_results['reference_value']
        offsets = analysis_results['offsets']
        
        print(f"\n🎯 OFFSET ANALYSIS (vs {ref_name}: {ref_value:.3f}m):")
        for name, offset_info in offsets.items():
            abs_offset = offset_info['absolute_offset']
            rel_offset = offset_info['relative_offset_percent']
            emoji = "🧭" if "INS" in name else "📏"
            print(f"   {emoji} {name}: {offset_info['measurement_value']:.3f}m (offset: {abs_offset:+.3f}m, {rel_offset:+.1f}%)")
    
    # Individual measurement details
    if distance_data:
        print(f"\n📏 INDIVIDUAL MEASUREMENTS:")
        for name, data in distance_data.items():
            emoji = "🧭" if "INS" in name else "🎯" if "Navigation" in name else "🎮" if any(x in name for x in ['Guidance', 'Desired']) else "📏"
            desc = data.get('description', 'No description')
            print(f"   {emoji} {name}: {data['value']:.3f}m")
            print(f"     └─ {desc}")
    
    print(f"\n✅ Analysis complete!")


def create_comprehensive_sonar_visualization(
    target_bag: str,
    frame_idx: int,
    rmax: float = None,
    exports_folder: str = "/Users/eirikvarnes/code/SOLAQUA/exports",
    figsize: Tuple[int, int] = (16, 12),
    analysis_tolerance: float = 0.1,
    show_analysis_report: bool = True,
    include_net_angle: bool = True
):
    """
    Create comprehensive sonar visualization with full analysis and net angle support.
    
    Args:
        target_bag: Bag identifier
        frame_idx: Frame index to visualize
        rmax: Maximum range (None for automatic detection)
        exports_folder: Path to exports folder
        figsize: Figure size
        analysis_tolerance: Tolerance for clustering analysis
        show_analysis_report: Whether to print analysis report
        include_net_angle: Whether to include net heading angle visualization
        
    Returns:
        Tuple of (matplotlib Figure, analysis results dict)
    """
    
    print(f"🎨 CREATING COMPREHENSIVE SONAR VISUALIZATION")
    print(f"   🎯 Target: {target_bag}, Frame: {frame_idx}")
    print(f"   📏 rmax: {rmax if rmax else 'auto-detect'}")
    print(f"   🧭 Net angle: {'enabled' if include_net_angle else 'disabled'}")
    print("=" * 60)
    
    # Extract sonar data
    raw_matrix, cone, extent, sonar_params = extract_raw_sonar_data_with_configurable_rmax(
        target_bag, frame_idx, rmax, exports_folder
    )
    
    if cone is None:
        print(f"❌ Failed to extract sonar data")
        return None, None
    
    # Load distance data
    raw_data, distance_measurements = load_all_distance_data_for_bag(target_bag, exports_folder)
    
    # Get timestamp for synchronization
    sonar_csv_file = Path(exports_folder) / "by_bag" / f"sensor_sonoptix_echo_image__{target_bag}_video.csv"
    sonar_df = pd.read_csv(sonar_csv_file)
    sonar_timestamp = pd.to_datetime(sonar_df.loc[frame_idx, 'ts_utc'])
    
    # Extract net heading angle if enabled
    net_angle_rad = 0.0
    net_angle_deg = 0.0
    angle_source = "default (disabled)"
    
    if include_net_angle:
        net_angle_rad, net_angle_deg, angle_source, time_offset = extract_net_heading_at_timestamp(
            sonar_timestamp, raw_data.get('navigation')
        )
        if abs(net_angle_deg) > 0.1:
            print(f"🧭 Net heading: {net_angle_deg:.1f}° ({angle_source})")
    
    # Collect synchronized measurements
    distance_data = collect_distance_measurements_at_timestamp(
        sonar_timestamp, raw_data.get('navigation'), raw_data.get('guidance'), distance_measurements
    )
    
    print(f"📊 Found {len(distance_data)} synchronized measurements")
    
    # Create visualization with angle support
    fig = create_enhanced_sonar_plot_with_measurements(
        cone, extent, distance_data, sonar_params,
        frame_idx=frame_idx,
        timestamp=sonar_timestamp.strftime('%H:%M:%S'),
        figsize=figsize,
        net_angle_rad=net_angle_rad,
        angle_source=angle_source,
        title_prefix=f"Comprehensive Sonar Analysis - {target_bag}"
    )
    
    # Perform analysis
    analysis_results = analyze_distance_measurements_clustering(
        distance_data, tolerance=analysis_tolerance,
        reference_measurement='Navigation NetDistance' if 'Navigation NetDistance' in distance_data else None
    )
    
    # Add angle information to results
    analysis_results['net_angle_rad'] = net_angle_rad
    analysis_results['net_angle_deg'] = net_angle_deg
    analysis_results['angle_source'] = angle_source
    
    # Print analysis report
    if show_analysis_report and distance_data:
        print_distance_analysis_report(analysis_results, distance_data)
    
    print(f"\n✅ Comprehensive visualization complete!")
    
    return fig, analysis_results
