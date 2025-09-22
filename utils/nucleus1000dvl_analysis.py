# Nucleus1000DVL Data Analysis and Visualization Utilities
# ========================================================
# Extract and visualize DVL (Doppler Velocity Log) sensor data from SOLAQUA CSV exports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from datetime import datetime

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
        
        if interactive:
            self._plot_bottomtrack_interactive(data, bags_to_plot, plot_title)
        else:
            self._plot_bottomtrack_matplotlib(data, bags_to_plot, plot_title)
    
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
    
    def _plot_bottomtrack_matplotlib(self, data, bags_to_plot, title):
        """Create matplotlib bottomtrack velocity plot"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(title, fontsize=14)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(bags_to_plot)))
        
        for i, bag in enumerate(bags_to_plot):
            df = data[bag]
            if df is None or 'dvl_velocity_xyz.x' not in df.columns:
                continue
            
            # Filter valid data
            valid_mask = df['data_valid'] == True if 'data_valid' in df.columns else pd.Series([True] * len(df))
            df_valid = df[valid_mask]
            
            if len(df_valid) == 0:
                continue
            
            color = colors[i]
            
            # Plot velocities
            axes[0].plot(df_valid['t_rel_min'], df_valid['dvl_velocity_xyz.x'], 
                        color=color, label=f"{bag}", alpha=0.7)
            axes[1].plot(df_valid['t_rel_min'], df_valid['dvl_velocity_xyz.y'], 
                        color=color, alpha=0.7)
            axes[2].plot(df_valid['t_rel_min'], df_valid['dvl_velocity_xyz.z'], 
                        color=color, alpha=0.7)
        
        # Formatting
        axes[0].set_ylabel('X Velocity (m/s)')
        axes[1].set_ylabel('Y Velocity (m/s)')
        axes[2].set_ylabel('Z Velocity (m/s)')
        axes[2].set_xlabel('Time (minutes)')
        
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
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
        
        if interactive:
            self._plot_ins_interactive(df, variables, title)
        else:
            self._plot_ins_matplotlib(df, variables, title)
    
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
    
    def _plot_ins_matplotlib(self, df, variables, title):
        """Create matplotlib INS plot"""
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=14)
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            if var == 'quaternion' and all(col in df.columns for col in ['quaternion.x', 'quaternion.y', 'quaternion.z', 'quaternion.w']):
                ax.plot(df['t_rel_min'], df['quaternion.x'], 'r-', label='qx', alpha=0.7)
                ax.plot(df['t_rel_min'], df['quaternion.y'], 'g-', label='qy', alpha=0.7)
                ax.plot(df['t_rel_min'], df['quaternion.z'], 'b-', label='qz', alpha=0.7)
                ax.plot(df['t_rel_min'], df['quaternion.w'], 'orange', label='qw', alpha=0.7)
                ax.set_ylabel('Quaternion')
                
            elif var == 'position' and all(col in df.columns for col in ['positionFrame.x', 'positionFrame.y', 'positionFrame.z']):
                ax.plot(df['t_rel_min'], df['positionFrame.x'], 'r-', label='x', alpha=0.7)
                ax.plot(df['t_rel_min'], df['positionFrame.y'], 'g-', label='y', alpha=0.7)
                ax.plot(df['t_rel_min'], df['positionFrame.z'], 'b-', label='z', alpha=0.7)
                ax.set_ylabel('Position (m)')
                
            elif var == 'velocity' and all(col in df.columns for col in ['velocityNed.x', 'velocityNed.y', 'velocityNed.z']):
                ax.plot(df['t_rel_min'], df['velocityNed.x'], 'r-', label='North', alpha=0.7)
                ax.plot(df['t_rel_min'], df['velocityNed.y'], 'g-', label='East', alpha=0.7)
                ax.plot(df['t_rel_min'], df['velocityNed.z'], 'b-', label='Down', alpha=0.7)
                ax.set_ylabel('Velocity (m/s)')
                
            elif var == 'turnrate' and all(col in df.columns for col in ['turnRate.x', 'turnRate.y', 'turnRate.z']):
                ax.plot(df['t_rel_min'], df['turnRate.x'], 'r-', label='roll rate', alpha=0.7)
                ax.plot(df['t_rel_min'], df['turnRate.y'], 'g-', label='pitch rate', alpha=0.7)
                ax.plot(df['t_rel_min'], df['turnRate.z'], 'b-', label='yaw rate', alpha=0.7)
                ax.set_ylabel('Turn Rate (deg/s)')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(var.title())
        
        axes[-1].set_xlabel('Time (minutes)')
        plt.tight_layout()
        plt.show()
    
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
        
        if interactive:
            self._plot_trajectory_2d_interactive(data, bags_to_plot, title)
        else:
            self._plot_trajectory_2d_matplotlib(data, bags_to_plot, title)
    
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
    
    def _plot_trajectory_2d_matplotlib(self, data, bags_to_plot, title):
        """Create matplotlib 2D trajectory plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(bags_to_plot)))
        
        for i, bag in enumerate(bags_to_plot):
            df = data[bag]
            if df is None or 'positionFrame.x' not in df.columns:
                continue
            
            color = colors[i]
            
            # Plot trajectory
            ax.plot(df['positionFrame.x'], df['positionFrame.y'], 
                   color=color, label=bag, alpha=0.7, linewidth=2)
            
            # Add start and end markers
            if len(df) > 0:
                ax.scatter(df['positionFrame.x'].iloc[0], df['positionFrame.y'].iloc[0], 
                          color=color, s=100, marker='s', alpha=0.8, edgecolor='black')
                ax.scatter(df['positionFrame.x'].iloc[-1], df['positionFrame.y'].iloc[-1], 
                          color=color, s=100, marker='D', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()
    
    def export_data_summary(self, output_file="nucleus1000dvl_summary.csv"):
        """
        Export a summary of all available data to CSV
        
        Parameters:
        -----------
        output_file : str
            Output CSV filename
        """
        summary_data = []
        
        for sensor in self.available_sensors:
            for bag in self.available_bags:
                data = self._load_single_file(f"nucleus1000dvl_{sensor}", bag, verbose=False)
                
                if data is not None:
                    summary_data.append({
                        'sensor': sensor,
                        'bag': bag,
                        'samples': len(data),
                        'duration_min': data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0,
                        'start_time': data['datetime'].min() if 'datetime' in data.columns else None,
                        'end_time': data['datetime'].max() if 'datetime' in data.columns else None,
                        'columns': len(data.columns)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to exports/outputs folder
        output_path = Path("exports/outputs") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_df.to_csv(output_path, index=False)
        print(f"✅ Summary exported to: {output_path}")
        
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
    
    def _compare_dvl_matplotlib(self, nucleus_bt, nucleus_ins, sensor_pos, sensor_vel, bag_name):
        """Create matplotlib DVL comparison plot"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
        fig.suptitle(f'DVL Sensor Comparison - {bag_name}', fontsize=14)
        
        # Velocity comparisons
        if nucleus_bt is not None and 'dvl_velocity_xyz.x' in nucleus_bt.columns:
            valid_mask = nucleus_bt.get('data_valid', pd.Series([True] * len(nucleus_bt))) == True
            nucleus_valid = nucleus_bt[valid_mask]
            
            if len(nucleus_valid) > 0:
                axes[0,0].plot(nucleus_valid['t_rel_min'], nucleus_valid['dvl_velocity_xyz.x'], 
                             'r-', label='Nucleus1000', alpha=0.7)
                axes[0,1].plot(nucleus_valid['t_rel_min'], nucleus_valid['dvl_velocity_xyz.y'], 
                             'r-', alpha=0.7)
                axes[1,0].plot(nucleus_valid['t_rel_min'], nucleus_valid['dvl_velocity_xyz.z'], 
                             'r-', alpha=0.7)
        
        if sensor_vel is not None and 'velocity.x' in sensor_vel.columns:
            axes[0,0].plot(sensor_vel['t_rel_min'], sensor_vel['velocity.x'], 
                         'b--', label='Sensor DVL', alpha=0.7)
            axes[0,1].plot(sensor_vel['t_rel_min'], sensor_vel['velocity.y'], 
                         'b--', alpha=0.7)
            axes[1,0].plot(sensor_vel['t_rel_min'], sensor_vel['velocity.z'], 
                         'b--', alpha=0.7)
        
        # Position comparisons
        if nucleus_ins is not None and 'positionFrame.x' in nucleus_ins.columns:
            axes[1,1].plot(nucleus_ins['t_rel_min'], nucleus_ins['positionFrame.x'], 
                         'g-', label='Nucleus1000', alpha=0.7)
            axes[2,0].plot(nucleus_ins['t_rel_min'], nucleus_ins['positionFrame.y'], 
                         'g-', alpha=0.7)
            axes[2,1].plot(nucleus_ins['t_rel_min'], nucleus_ins['positionFrame.z'], 
                         'purple', alpha=0.7)
        
        if sensor_pos is not None and 'x' in sensor_pos.columns:
            axes[1,1].plot(sensor_pos['t_rel_min'], sensor_pos['x'], 
                         'orange', linestyle='--', label='Sensor DVL', alpha=0.7)
            axes[2,0].plot(sensor_pos['t_rel_min'], sensor_pos['y'], 
                         'orange', linestyle='--', alpha=0.7)
            axes[2,1].plot(sensor_pos['t_rel_min'], sensor_pos['z'], 
                         'purple', linestyle='--', alpha=0.7)
        
        # Format subplots
        axes[0,0].set_title('X Velocity')
        axes[0,0].set_ylabel('Velocity (m/s)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].set_title('Y Velocity')
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].set_title('Z Velocity')
        axes[1,0].set_ylabel('Velocity (m/s)')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].set_title('X Position')
        axes[1,1].set_ylabel('Position (m)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        axes[2,0].set_title('Y Position')
        axes[2,0].set_ylabel('Position (m)')
        axes[2,0].set_xlabel('Time (minutes)')
        axes[2,0].grid(True, alpha=0.3)
        
        axes[2,1].set_title('Z Position')
        axes[2,1].set_ylabel('Position (m)')
        axes[2,1].set_xlabel('Time (minutes)')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_data_summary(self, output_file="nucleus1000dvl_summary.csv"):
        """
        Export a summary of all available data to CSV
        
        Parameters:
        -----------
        output_file : str
            Output CSV filename
        """
        summary_data = []
        
        for sensor in self.available_sensors:
            for bag in self.available_bags:
                data = self._load_single_file(f"nucleus1000dvl_{sensor}", bag, verbose=False)
                
                if data is not None:
                    summary_data.append({
                        'sensor': sensor,
                        'bag': bag,
                        'samples': len(data),
                        'duration_min': data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0,
                        'start_time': data['datetime'].min() if 'datetime' in data.columns else None,
                        'end_time': data['datetime'].max() if 'datetime' in data.columns else None,
                        'columns': len(data.columns)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to exports/outputs folder
        output_path = Path("exports/outputs") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_df.to_csv(output_path, index=False)
        print(f"✅ Summary exported to: {output_path}")
        
        return summary_df


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


# Convenience functions
def quick_load_dvl_data(by_bag_folder="exports/by_bag"):
    """Quick function to create and return analyzer"""
    return Nucleus1000DVLAnalyzer(by_bag_folder)


def quick_bottomtrack_plot(bag_name=None, by_bag_folder="exports/by_bag"):
    """Quick function to plot bottomtrack data"""
    analyzer = Nucleus1000DVLAnalyzer(by_bag_folder)
    analyzer.plot_bottomtrack_velocity(bag_name)
    return analyzer


def quick_trajectory_plot(bag_name=None, by_bag_folder="exports/by_bag"):
    """Quick function to plot trajectory"""
    analyzer = Nucleus1000DVLAnalyzer(by_bag_folder)
    analyzer.plot_trajectory_2d(bag_name)
    return analyzer
