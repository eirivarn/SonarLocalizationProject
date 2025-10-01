# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

# Nucleus1000DVL Data Analysis and Visualization Utilities
# ========================================================
# Extract and visualize DVL (Doppler Velocity Log) sensor data from SOLAQUA CSV exports

import pandas as pd
import numpy as np
from pathlib import Path

# Check for optional libraries
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class Nucleus1000DVLAnalyzer:
    """
    Main class for analyzing Nucleus1000DVL data from CSV files
    """
    
    def __init__(self, by_bag_folder="exports/by_bag"):
        """
        Initialize analyzer with path to by_bag folder
        
        Parameters:
        -----------
        by_bag_folder : str
            Path to folder containing CSV files
        """
        self.by_bag_folder = Path(by_bag_folder)
        self.data = {}
        self.available_bags = []
        self.available_sensors = []
        
        if self.by_bag_folder.exists():
            self._discover_files()
        else:
            print(f"⚠️  Warning: {by_bag_folder} does not exist")
    
    def _discover_files(self):
        """Discover available nucleus1000dvl and sensor_dvl files"""
        # Find nucleus1000dvl files
        nucleus_files = list(self.by_bag_folder.glob("nucleus1000dvl_*.csv"))
        
        # Find sensor_dvl files
        sensor_dvl_files = list(self.by_bag_folder.glob("sensor_dvl_*.csv"))
        
        # Extract unique bags and sensors
        bags = set()
        sensors = set()
        
        # Process nucleus1000dvl files
        for file in nucleus_files:
            parts = file.stem.split("__")
            if len(parts) >= 2:
                sensor_name = parts[0]
                bag_name = parts[1].replace("_data", "")
                
                sensors.add(sensor_name.replace("nucleus1000dvl_", ""))
                bags.add(bag_name)
        
        # Process sensor_dvl files  
        for file in sensor_dvl_files:
            parts = file.stem.split("__")
            if len(parts) >= 2:
                sensor_name = parts[0]
                bag_name = parts[1].replace("_data", "")
                
                sensors.add(sensor_name)
                bags.add(bag_name)
        
        self.available_bags = sorted(list(bags))
        self.available_sensors = sorted(list(sensors))
        
        print(f"🔍 Found DVL data:")
        print(f"   📅 Bags: {len(self.available_bags)}")
        print(f"   📊 Sensors: {len(self.available_sensors)}")
        print(f"   Sensors: {', '.join(self.available_sensors)}")
    
    def load_sensor_data(self, sensor_type, bag_name=None, verbose=True):
        """
        Load data for a specific sensor type and bag
        
        Parameters:
        -----------
        sensor_type : str
            Type of sensor (bottomtrack, ins, altimeter, imu, magnetometer, watertrack)
        bag_name : str or None
            Specific bag name, or None for all bags
        verbose : bool
            Print loading information
            
        Returns:
        --------
        pandas.DataFrame or dict : Loaded data
        """
        if sensor_type not in self.available_sensors:
            print(f"❌ Sensor '{sensor_type}' not available")
            print(f"   Available sensors: {', '.join(self.available_sensors)}")
            return None
        
        if bag_name is None:
            # Load all bags for this sensor
            all_data = {}
            for bag in self.available_bags:
                data = self._load_single_file(f"nucleus1000dvl_{sensor_type}", bag, verbose=verbose)
                if data is not None:
                    all_data[bag] = data
            
            if verbose:
                print(f"✅ Loaded {sensor_type} data for {len(all_data)} bags")
            
            return all_data
        else:
            # Load specific bag
            if bag_name not in self.available_bags:
                print(f"❌ Bag '{bag_name}' not available")
                print(f"   Available bags: {', '.join(self.available_bags)}")
                return None
            
            return self._load_single_file(f"nucleus1000dvl_{sensor_type}", bag_name, verbose=verbose)
    
    def _load_single_file(self, sensor_prefix, bag_name, verbose=True):
        """Load a single CSV file"""
        filename = f"{sensor_prefix}__{bag_name}_data.csv"
        filepath = self.by_bag_folder / filename
        
        if not filepath.exists():
            if verbose:
                print(f"⚠️  File not found: {filename}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Convert timestamps
            if 't' in df.columns:
                df['datetime'] = pd.to_datetime(df['t'], unit='s')
            if 't_rel' in df.columns:
                df['t_rel_min'] = df['t_rel'] / 60.0  # Convert to minutes
            
            # Special handling for sensor_dvl files
            if sensor_prefix.startswith('sensor_dvl'):
                # Parse complex beam data if present
                if 'beams' in df.columns:
                    try:
                        # Extract beam information (simplified parsing)
                        df['num_valid_beams'] = df['beams'].str.count('valid=True')
                    except:
                        pass
            
            if verbose:
                print(f"📁 Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def get_summary(self, sensor_type=None):
        """
        Get summary of available data
        
        Parameters:
        -----------
        sensor_type : str or None
            Specific sensor type, or None for all
        """
        print("📊 Nucleus1000DVL Data Summary")
        print("=" * 40)
        
        sensors_to_check = [sensor_type] if sensor_type else self.available_sensors
        
        for sensor in sensors_to_check:
            print(f"\n🔧 {sensor.upper()}:")
            
            for bag in self.available_bags:
                data = self._load_single_file(f"nucleus1000dvl_{sensor}", bag, verbose=False)
                if data is not None:
                    duration = data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0
                    print(f"   📅 {bag}: {len(data)} samples, {duration:.1f} min")
                else:
                    print(f"   📅 {bag}: No data")
    
    def plot_bottomtrack_velocity(self, bag_name=None, interactive=True):
        """
        Plot bottom track velocity data
        
        Parameters:
        -----------
        bag_name : str or None
            Specific bag or None for all
        interactive : bool
            Use plotly for interactive plots
        """
        data = self.load_sensor_data("bottomtrack", bag_name, verbose=False)
        
        if data is None:
            return
        
        # Handle single bag vs multiple bags
        if isinstance(data, dict):
            bags_to_plot = data.keys()
            plot_title = "Bottom Track Velocity - All Bags"
        else:
            bags_to_plot = [bag_name]
            data = {bag_name: data}
            plot_title = f"Bottom Track Velocity - {bag_name}"
        
        # Always use interactive Plotly plots (matplotlib support removed)
        if HAS_PLOTLY:
            self._plot_bottomtrack_interactive(data, bags_to_plot, plot_title)
        else:
            print("❌ Plotly not available. Install plotly to view interactive plots.")
    
    def _plot_bottomtrack_interactive(self, data, bags_to_plot, title):
        """Create interactive bottomtrack velocity plot with plotly"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['X Velocity', 'Y Velocity', 'Z Velocity'],
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, bag in enumerate(bags_to_plot):
            df = data[bag]
            if df is None or 'dvl_velocity_xyz.x' not in df.columns:
                continue
            
            color = colors[i % len(colors)]
            
            # Filter valid data
            valid_mask = df['data_valid'] == True if 'data_valid' in df.columns else pd.Series([True] * len(df))
            df_valid = df[valid_mask]
            
            if len(df_valid) == 0:
                continue
            
            # X velocity
            fig.add_trace(
                go.Scatter(
                    x=df_valid['t_rel_min'],
                    y=df_valid['dvl_velocity_xyz.x'],
                    name=f"{bag} - X",
                    line=dict(color=color),
                    legendgroup=bag
                ),
                row=1, col=1
            )
            
            # Y velocity
            fig.add_trace(
                go.Scatter(
                    x=df_valid['t_rel_min'],
                    y=df_valid['dvl_velocity_xyz.y'],
                    name=f"{bag} - Y",
                    line=dict(color=color, dash='dash'),
                    legendgroup=bag
                ),
                row=2, col=1
            )
            
            # Z velocity
            fig.add_trace(
                go.Scatter(
                    x=df_valid['t_rel_min'],
                    y=df_valid['dvl_velocity_xyz.z'],
                    name=f"{bag} - Z",
                    line=dict(color=color, dash='dot'),
                    legendgroup=bag
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
        fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
        fig.update_yaxes(title_text="Velocity (m/s)", row=3, col=1)
        
        fig.show()
    
    # Matplotlib-based plotting support removed. Use Plotly interactive helpers instead.
    
    def plot_ins_data(self, bag_name=None, variables=['quaternion', 'position', 'velocity'], interactive=True):
        """
        Plot INS (Inertial Navigation System) data
        
        Parameters:
        -----------
        bag_name : str or None
            Specific bag or None for first available
        variables : list
            Variables to plot ['quaternion', 'position', 'velocity', 'turnrate']
        interactive : bool
            Use plotly for interactive plots
        """
        data = self.load_sensor_data("ins", bag_name, verbose=False)
        
        if data is None:
            return
        
        # Handle single bag vs multiple bags
        if isinstance(data, dict):
            if bag_name is None:
                bag_name = list(data.keys())[0]  # Use first bag
            df = data[bag_name]
            title = f"INS Data - {bag_name}"
        else:
            df = data
            title = f"INS Data - {bag_name}"
        
        if df is None or len(df) == 0:
            print(f"❌ No INS data for {bag_name}")
            return
        
        # Always use interactive Plotly plots (matplotlib support removed)
        if HAS_PLOTLY:
            self._plot_ins_interactive(df, variables, title)
        else:
            print("❌ Plotly not available. Install plotly to view interactive plots.")
    
    def _plot_ins_interactive(self, df, variables, title):
        """Create interactive INS plot with plotly"""
        n_vars = len(variables)
        fig = make_subplots(
            rows=n_vars, cols=1,
            subplot_titles=[v.title() for v in variables],
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        for i, var in enumerate(variables):
            row = i + 1
            
            if var == 'quaternion' and all(col in df.columns for col in ['quaternion.x', 'quaternion.y', 'quaternion.z', 'quaternion.w']):
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['quaternion.x'], name='qx', line=dict(color='red')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['quaternion.y'], name='qy', line=dict(color='green')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['quaternion.z'], name='qz', line=dict(color='blue')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['quaternion.w'], name='qw', line=dict(color='orange')), row=row, col=1)
            
            elif var == 'position' and all(col in df.columns for col in ['positionFrame.x', 'positionFrame.y', 'positionFrame.z']):
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['positionFrame.x'], name='pos_x', line=dict(color='red')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['positionFrame.y'], name='pos_y', line=dict(color='green')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['positionFrame.z'], name='pos_z', line=dict(color='blue')), row=row, col=1)
            
            elif var == 'velocity' and all(col in df.columns for col in ['velocityNed.x', 'velocityNed.y', 'velocityNed.z']):
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['velocityNed.x'], name='vel_N', line=dict(color='red')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['velocityNed.y'], name='vel_E', line=dict(color='green')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['velocityNed.z'], name='vel_D', line=dict(color='blue')), row=row, col=1)
            
            elif var == 'turnrate' and all(col in df.columns for col in ['turnRate.x', 'turnRate.y', 'turnRate.z']):
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['turnRate.x'], name='turn_x', line=dict(color='red')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['turnRate.y'], name='turn_y', line=dict(color='green')), row=row, col=1)
                fig.add_trace(go.Scatter(x=df['t_rel_min'], y=df['turnRate.z'], name='turn_z', line=dict(color='blue')), row=row, col=1)
        
        fig.update_layout(title=title, height=200*n_vars + 100, hovermode='x unified')
        fig.update_xaxes(title_text="Time (minutes)", row=n_vars, col=1)
        fig.show()
    
    # Matplotlib-based plotting support removed. Use Plotly interactive helpers instead.
    
    def plot_trajectory_2d(self, bag_name=None, interactive=True):
        """
        Plot 2D trajectory from INS position data
        
        Parameters:
        -----------
        bag_name : str or None
            Specific bag or None for all bags
        interactive : bool
            Use plotly for interactive plots
        """
        data = self.load_sensor_data("ins", bag_name, verbose=False)
        
        if data is None:
            return
        
        # Handle single bag vs multiple bags
        if isinstance(data, dict):
            bags_to_plot = data.keys()
            title = "Vehicle Trajectory - All Bags"
        else:
            bags_to_plot = [bag_name]
            data = {bag_name: data}
            title = f"Vehicle Trajectory - {bag_name}"
        
        # Always use interactive Plotly plots (matplotlib support removed)
        if HAS_PLOTLY:
            self._plot_trajectory_2d_interactive(data, bags_to_plot, title)
        else:
            print("❌ Plotly not available. Install plotly to view interactive plots.")
    
    def _plot_trajectory_2d_interactive(self, data, bags_to_plot, title):
        """Create interactive 2D trajectory plot"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, bag in enumerate(bags_to_plot):
            df = data[bag]
            if df is None or 'positionFrame.x' not in df.columns:
                continue
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=df['positionFrame.x'],
                y=df['positionFrame.y'],
                mode='lines+markers',
                name=bag,
                line=dict(color=color),
                marker=dict(size=3)
            ))
            
            # Add start and end markers
            if len(df) > 0:
                # Start point
                fig.add_trace(go.Scatter(
                    x=[df['positionFrame.x'].iloc[0]],
                    y=[df['positionFrame.y'].iloc[0]],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='square'),
                    name=f"{bag} Start",
                    showlegend=False
                ))
                
                # End point
                fig.add_trace(go.Scatter(
                    x=[df['positionFrame.x'].iloc[-1]],
                    y=[df['positionFrame.y'].iloc[-1]],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='diamond'),
                    name=f"{bag} End",
                    showlegend=False
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
            height=600
        )
        
        fig.show()
    
    # Matplotlib-based plotting support removed. Use Plotly interactive helpers instead.

    # -------------------------
    # Notebook helper methods
    # -------------------------
    def explore_bag(self, bag_name):
        """
        Print a compact exploration summary for a single bag.

        This replicates the "BASIC DATA EXPLORATION" notebook cell
        and is useful for quickly inspecting available sensors for a bag.
        """
        if bag_name is None:
            print("❌ No bag specified for exploration")
            return

        print(f"\n🔍 Exploring bag: {bag_name}")
        for sensor in self.available_sensors:
            print(f"\n🔧 {sensor.upper()} Data:")
            data = self.load_sensor_data(sensor, bag_name, verbose=False)
            if data is not None and len(data) > 0:
                try:
                    duration_min = data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0
                except Exception:
                    duration_min = 0
                print(f"   📏 Shape: {data.shape}")
                print(f"   ⏱️  Duration: {duration_min:.1f} minutes")
            else:
                print("   ❌ No data available")

    def compare_bottomtrack_across_bags(self, output_folder="exports/outputs", export_plots=False):
        """
        Create comparison plots for bottomtrack velocity across all available bags.

        This encapsulates the multi-file comparison notebook cell.
        """
        if 'bottomtrack' not in self.available_sensors or len(self.available_bags) <= 1:
            print("⚠️  Not enough bottomtrack data across bags for comparison")
            return

        print(f"📊 Comparing bottomtrack across {len(self.available_bags)} bags")
        # Delegate to the interactive bottomtrack plot (handles multi-bag)
        if HAS_PLOTLY:
            try:
                self.plot_bottomtrack_velocity(None, interactive=True)
            except Exception as e:
                print(f"⚠️ Failed to create interactive bottomtrack comparison: {e}")
        else:
            print("❌ Plotly not available. Install plotly to view interactive comparisons.")

    def compute_summary_stats(self, export_summary=True, output_folder="exports/outputs"):
        """
        Compute the statistical summary used in the notebook and optionally export it.

        Returns a pandas DataFrame with summary rows.
        """
        summary_stats = []
        for sensor in self.available_sensors:
            for bag in self.available_bags:
                data = self._load_single_file(f"nucleus1000dvl_{sensor}", bag, verbose=False)
                if data is None or len(data) == 0:
                    continue
                duration_min = data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0
                sample_rate = len(data) / data['t_rel'].max() if 't_rel' in data.columns and data['t_rel'].max() > 0 else 0
                entry = {'sensor': sensor, 'bag': bag, 'samples': len(data), 'duration_min': duration_min, 'sample_rate_hz': sample_rate}

                # Sensor-specific enrichments (keep lightweight)
                if sensor == 'bottomtrack' and 'dvl_velocity_xyz.x' in data.columns:
                    valid_mask = data.get('data_valid', pd.Series([True] * len(data))) == True
                    data_valid = data[valid_mask]
                    if len(data_valid) > 0:
                        speed = np.sqrt(data_valid['dvl_velocity_xyz.x']**2 + data_valid['dvl_velocity_xyz.y']**2 + data_valid['dvl_velocity_xyz.z']**2)
                        entry.update({'mean_speed': speed.mean(), 'std_speed': speed.std(), 'valid_percent': len(data_valid)/len(data)*100})

                summary_stats.append(entry)

        summary_df = pd.DataFrame(summary_stats)
        if not summary_df.empty and export_summary:
            outp = Path(output_folder) / "nucleus1000dvl_detailed_summary.csv"
            outp.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(outp, index=False)
            print(f"💾 Summary exported to: {outp}")

        return summary_df

    def run_navigation_guidance_analysis(self, bag_name, interactive=True):
        """
        Run the navigation/guidance notebook section via utils.navigation_guidance_analysis.
        """
        try:
            import importlib
            import utils.navigation_guidance_analysis as nav_guidance_utils
            importlib.reload(nav_guidance_utils)

            nav_analyzer = nav_guidance_utils.NavigationGuidanceAnalyzer(self.by_bag_folder)
            print(f"📁 Running navigation/guidance analysis for bag: {bag_name}")
            nav_analyzer.analyze_guidance_errors(bag_name, interactive=interactive)
            nav_analyzer.analyze_navigation_plane(bag_name, interactive=interactive)
            nav_analyzer.compare_navigation_guidance(bag_name, interactive=interactive)
            return True
        except Exception as e:
            print(f"⚠️ Navigation/guidance analysis failed: {e}")
            return False
 
    def compare_dvl_sensors(self, bag_name=None, interactive=True):
        """
        Compare nucleus1000dvl vs sensor_dvl position and velocity data
        
        Parameters:
        -----------
        bag_name : str or None
            Specific bag name or None for first available
        interactive : bool
            Use plotly for interactive plots
        """
        if bag_name is None:
            if not self.available_bags:
                print("❌ No bags available")
                return
            bag_name = self.available_bags[0]
            print(f"📊 Using bag: {bag_name}")
        
        print(f"🔄 Comparing DVL sensors for bag: {bag_name}")
        
        # Load nucleus1000dvl data
        nucleus_bt = self.load_sensor_data("bottomtrack", bag_name, verbose=False)
        nucleus_ins = self.load_sensor_data("ins", bag_name, verbose=False)
        
        # Load sensor_dvl data
        sensor_pos = self.load_sensor_data("sensor_dvl_position", bag_name, verbose=False)
        sensor_vel = self.load_sensor_data("sensor_dvl_velocity", bag_name, verbose=False)
        
        # Handle dict returns
        if isinstance(nucleus_bt, dict):
            nucleus_bt = nucleus_bt.get(bag_name)
        if isinstance(nucleus_ins, dict):
            nucleus_ins = nucleus_ins.get(bag_name)
        if isinstance(sensor_pos, dict):
            sensor_pos = sensor_pos.get(bag_name)
        if isinstance(sensor_vel, dict):
            sensor_vel = sensor_vel.get(bag_name)
        
        if interactive and HAS_PLOTLY:
            self._compare_dvl_interactive(nucleus_bt, nucleus_ins, sensor_pos, sensor_vel, bag_name)
        else:
            self._compare_dvl_matplotlib(nucleus_bt, nucleus_ins, sensor_pos, sensor_vel, bag_name)
    
    def _compare_dvl_interactive(self, nucleus_bt, nucleus_ins, sensor_pos, sensor_vel, bag_name):
        """Create interactive DVL comparison plot"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Velocity Comparison (X)', 'Velocity Comparison (Y)', 
                          'Velocity Comparison (Z)', 'Position Comparison (X)',
                          'Position Comparison (Y)', 'Altitude/Z Comparison'],
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # Velocity comparisons
        if nucleus_bt is not None and 'dvl_velocity_xyz.x' in nucleus_bt.columns:
            valid_mask = nucleus_bt.get('data_valid', pd.Series([True] * len(nucleus_bt))) == True
            nucleus_valid = nucleus_bt[valid_mask]
            
            if len(nucleus_valid) > 0:
                # X velocity
                fig.add_trace(go.Scatter(
                    x=nucleus_valid['t_rel_min'], y=nucleus_valid['dvl_velocity_xyz.x'],
                    name='Nucleus1000 VX', line=dict(color='red')
                ), row=1, col=1)
                
                # Y velocity
                fig.add_trace(go.Scatter(
                    x=nucleus_valid['t_rel_min'], y=nucleus_valid['dvl_velocity_xyz.y'],
                    name='Nucleus1000 VY', line=dict(color='red')
                ), row=1, col=2)
                
                # Z velocity
                fig.add_trace(go.Scatter(
                    x=nucleus_valid['t_rel_min'], y=nucleus_valid['dvl_velocity_xyz.z'],
                    name='Nucleus1000 VZ', line=dict(color='red')
                ), row=2, col=1)
        
        if sensor_vel is not None and 'velocity.x' in sensor_vel.columns:
            # X velocity
            fig.add_trace(go.Scatter(
                x=sensor_vel['t_rel_min'], y=sensor_vel['velocity.x'],
                name='Sensor DVL VX', line=dict(color='blue', dash='dash')
            ), row=1, col=1)
            
            # Y velocity
            fig.add_trace(go.Scatter(
                x=sensor_vel['t_rel_min'], y=sensor_vel['velocity.y'],
                name='Sensor DVL VY', line=dict(color='blue', dash='dash')
            ), row=1, col=2)
            
            # Z velocity
            fig.add_trace(go.Scatter(
                x=sensor_vel['t_rel_min'], y=sensor_vel['velocity.z'],
                name='Sensor DVL VZ', line=dict(color='blue', dash='dash')
            ), row=2, col=1)
        
        # Position comparisons
        if nucleus_ins is not None and 'positionFrame.x' in nucleus_ins.columns:
            # X position
            fig.add_trace(go.Scatter(
                x=nucleus_ins['t_rel_min'], y=nucleus_ins['positionFrame.x'],
                name='Nucleus1000 PosX', line=dict(color='green')
            ), row=2, col=2)
            
            # Y position
            fig.add_trace(go.Scatter(
                x=nucleus_ins['t_rel_min'], y=nucleus_ins['positionFrame.y'],
                name='Nucleus1000 PosY', line=dict(color='green')
            ), row=3, col=1)
        
        if sensor_pos is not None and 'x' in sensor_pos.columns:
            # X position
            fig.add_trace(go.Scatter(
                x=sensor_pos['t_rel_min'], y=sensor_pos['x'],
                name='Sensor DVL PosX', line=dict(color='orange', dash='dash')
            ), row=2, col=2)
            
            # Y position
            fig.add_trace(go.Scatter(
                x=sensor_pos['t_rel_min'], y=sensor_pos['y'],
                name='Sensor DVL PosY', line=dict(color='orange', dash='dash')
            ), row=3, col=1)
            
            # Z position comparison
            fig.add_trace(go.Scatter(
                x=sensor_pos['t_rel_min'], y=sensor_pos['z'],
                name='Sensor DVL PosZ', line=dict(color='purple', dash='dash')
            ), row=3, col=2)
        
        if nucleus_ins is not None and 'positionFrame.z' in nucleus_ins.columns:
            fig.add_trace(go.Scatter(
                x=nucleus_ins['t_rel_min'], y=nucleus_ins['positionFrame.z'],
                name='Nucleus1000 PosZ', line=dict(color='purple')
            ), row=3, col=2)
        
        fig.update_layout(
            title=f"DVL Sensor Comparison - {bag_name}",
            height=900,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
        fig.update_xaxes(title_text="Time (minutes)", row=3, col=2)
        
        fig.show()
    
    # Matplotlib-based DVL comparison removed. Use _compare_dvl_interactive instead.
    

def create_comprehensive_dvl_dashboard(analyzer, bag_name=None, output_html="dvl_dashboard.html"):
    """
    Create a comprehensive dashboard with all DVL sensor data
    
    Parameters:
    -----------
    analyzer : Nucleus1000DVLAnalyzer
        Initialized analyzer object
    bag_name : str or None
        Specific bag name or None for first available
    output_html : str
        Output HTML filename
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    
    # Determine which bag to use
    if bag_name is None:
        if not analyzer.available_bags:
            print("❌ No bags available")
            return
        bag_name = analyzer.available_bags[0]
        print(f"📊 Using bag: {bag_name}")
    
    # Load all sensor data for the bag
    sensor_data = {}
    for sensor in analyzer.available_sensors:
        data = analyzer.load_sensor_data(sensor, bag_name, verbose=False)
        if isinstance(data, dict):
            sensor_data[sensor] = data.get(bag_name)
        else:
            sensor_data[sensor] = data
    
    # Create dashboard with subplots
    n_sensors = len([s for s in sensor_data.values() if s is not None])
    
    fig = make_subplots(
        rows=n_sensors, cols=1,
        subplot_titles=[f"{sensor.upper()}" for sensor, data in sensor_data.items() if data is not None],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    row = 1
    for sensor, data in sensor_data.items():
        if data is None or len(data) == 0:
            continue
        
        if sensor == 'bottomtrack' and 'dvl_velocity_xyz.x' in data.columns:
            # Bottom track velocity
            valid_mask = data['data_valid'] == True if 'data_valid' in data.columns else pd.Series([True] * len(data))
            data_valid = data[valid_mask]
            
            if len(data_valid) > 0:
                fig.add_trace(go.Scatter(x=data_valid['t_rel_min'], y=data_valid['dvl_velocity_xyz.x'], 
                                       name='VX', line=dict(color='red')), row=row, col=1)
                fig.add_trace(go.Scatter(x=data_valid['t_rel_min'], y=data_valid['dvl_velocity_xyz.y'], 
                                       name='VY', line=dict(color='green')), row=row, col=1)
                fig.add_trace(go.Scatter(x=data_valid['t_rel_min'], y=data_valid['dvl_velocity_xyz.z'], 
                                       name='VZ', line=dict(color='blue')), row=row, col=1)
        
        elif sensor == 'ins' and 'positionFrame.x' in data.columns:
            # INS position
            fig.add_trace(go.Scatter(x=data['t_rel_min'], y=data['positionFrame.x'], 
                                   name='POS_X', line=dict(color='red')), row=row, col=1)
            fig.add_trace(go.Scatter(x=data['t_rel_min'], y=data['positionFrame.y'], 
                                   name='POS_Y', line=dict(color='green')), row=row, col=1)
            fig.add_trace(go.Scatter(x=data['t_rel_min'], y=data['positionFrame.z'], 
                                   name='POS_Z', line=dict(color='blue')), row=row, col=1)
        
        elif sensor == 'altimeter' and 'altimeter_distance' in data.columns:
            # Altimeter
            fig.add_trace(go.Scatter(x=data['t_rel_min'], y=data['altimeter_distance'], 
                                   name='ALT', line=dict(color='purple')), row=row, col=1)
        
        elif sensor == 'imu' and 'linear_acceleration.x' in data.columns:
            # IMU acceleration
            fig.add_trace(go.Scatter(x=data['t_rel_min'], y=data['linear_acceleration.x'], 
                                   name='ACC_X', line=dict(color='red')), row=row, col=1)
            fig.add_trace(go.Scatter(x=data['t_rel_min'], y=data['linear_acceleration.y'], 
                                   name='ACC_Y', line=dict(color='green')), row=row, col=1)
            fig.add_trace(go.Scatter(x=data['t_rel_min'], y=data['linear_acceleration.z'], 
                                   name='ACC_Z', line=dict(color='blue')), row=row, col=1)
        
        row += 1
    
    # Update layout
    fig.update_layout(
        title=f"Nucleus1000DVL Dashboard - {bag_name}",
        height=300 * n_sensors,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time (minutes)", row=n_sensors, col=1)
    
    # Save to HTML
    output_path = Path("exports/outputs") / output_html
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pyo.plot(fig, filename=str(output_path), auto_open=False)
    print(f"📊 Dashboard saved to: {output_path}")
    
    return fig


def run_full_notebook_workflow(by_bag_folder="exports/by_bag", bag_selection=None,
                               sensor_selection=None, export_summary=True, export_plots=False,
                               output_folder="exports/outputs", interactive=HAS_PLOTLY,
                               plot_style=None):
    """
    Run the main analysis workflow that the notebook performs.

    This replicates the high-level steps found in the notebook:
    - initialize analyzer
    - discover available bags/sensors and print summary
    - perform basic exploration for a selected bag
    - create various plots (bottomtrack, trajectories, INS, multi-file comparisons)
    - compute summary statistics and optionally export
    - create dashboard (interactive if Plotly available)
    - run DVL sensor comparison

    Parameters mirror the top-level notebook configuration so the notebook
    can call this single function.
    """
    print("🚀 Running full Nucleus1000DVL notebook workflow...")
    analyzer = Nucleus1000DVLAnalyzer(by_bag_folder)

    print(f"\n📋 Data Discovery Summary:")
    print(f"   📅 Available bags: {len(analyzer.available_bags)}")
    print(f"   🔧 Available sensors: {len(analyzer.available_sensors)}")

    # Choose bag
    selected_bag = bag_selection
    if selected_bag is None and analyzer.available_bags:
        selected_bag = analyzer.available_bags[0]

    if selected_bag is None:
        print("❌ No bag available. Aborting workflow.")
        return {'error': 'no_bag'}

    print(f"\n📅 Using bag: {selected_bag}")

    # Basic exploration
    print("\n📊 Basic data exploration for selected bag")
    for sensor in analyzer.available_sensors:
        print(f"\n🔧 {sensor.upper()} Data:")
        data = analyzer.load_sensor_data(sensor, selected_bag, verbose=False)
        if data is not None and len(data) > 0:
            try:
                duration_min = data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0
            except Exception:
                duration_min = 0
            print(f"   📏 Shape: {data.shape}")
            print(f"   ⏱️  Duration: {duration_min:.1f} minutes")
        else:
            print("   ❌ No data available")

    # Generate targeted plots (non-interactive by default unless interactive True)
    print("\n🚀 Generating plots (may open interactive windows if enabled)...")
    # Bottomtrack
    if 'bottomtrack' in analyzer.available_sensors:
        try:
            analyzer.plot_bottomtrack_velocity(selected_bag if selected_bag else None, interactive=interactive)
        except Exception as e:
            print(f"⚠️ Failed to plot bottomtrack: {e}")

    # Trajectory
    if 'ins' in analyzer.available_sensors:
        try:
            analyzer.plot_trajectory_2d(selected_bag if selected_bag else None, interactive=interactive)
        except Exception as e:
            print(f"⚠️ Failed to plot trajectory: {e}")

    # INS data
    try:
        analyzer.plot_ins_data(selected_bag if selected_bag else None, variables=['position', 'velocity'], interactive=interactive)
    except Exception as e:
        print(f"⚠️ Failed to plot INS: {e}")

    # Multi-file comparison across bags (if multiple)
    if len(analyzer.available_bags) > 1:
        try:
            # Use existing quick plotting functions in the analyzer
            print("\n📈 Running multi-file comparisons across bags...")
            # Use the analyzer methods directly where useful
            # The original notebook produced bottomtrack comparison; reuse that
            analyzer.plot_bottomtrack_velocity(None, interactive=interactive)
        except Exception as e:
            print(f"⚠️ Multi-file comparison failed: {e}")

    # Statistical summary and export
    try:
        print("\n📊 Computing summary statistics...")
        # Reuse notebook-style summary generation
        summary_stats = []
        for sensor in analyzer.available_sensors:
            for bag in analyzer.available_bags:
                data = analyzer._load_single_file(f"nucleus1000dvl_{sensor}", bag, verbose=False)
                if data is not None and len(data) > 0:
                    duration_min = data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0
                    sample_rate = len(data) / data['t_rel'].max() if 't_rel' in data.columns and data['t_rel'].max() > 0 else 0
                    entry = {'sensor': sensor, 'bag': bag, 'samples': len(data), 'duration_min': duration_min, 'sample_rate_hz': sample_rate}
                    summary_stats.append(entry)
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            print(f"\n📋 Overall Summary: {len(summary_df)} datasets")
            if export_summary:
                outp = Path(output_folder) / "nucleus1000dvl_detailed_summary.csv"
                outp.parent.mkdir(parents=True, exist_ok=True)
                summary_df.to_csv(outp, index=False)
                print(f"💾 Summary exported to: {outp}")
    except Exception as e:
        print(f"⚠️ Summary generation failed: {e}")

    # Dashboard creation
    try:
        if interactive and HAS_PLOTLY:
            print("\n🎛️ Creating interactive dashboard...")
            create_comprehensive_dvl_dashboard(analyzer, bag_name=selected_bag, output_html="nucleus1000dvl_dashboard.html")
        else:
            print("\nℹ️ Skipping interactive dashboard (Plotly not available or interactive=False).")
    except Exception as e:
        print(f"⚠️ Dashboard creation failed: {e}")

    # DVL sensor comparison
    try:
        print("\n🔍 Running DVL sensor comparison...")
        analyzer.compare_dvl_sensors(selected_bag, interactive=interactive)
    except Exception as e:
        print(f"⚠️ DVL sensor comparison failed: {e}")

    print("\n✅ Notebook workflow complete.")
    return {'status': 'done', 'selected_bag': selected_bag}
