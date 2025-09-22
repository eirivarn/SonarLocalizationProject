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
            'rmax': 30.0,
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
