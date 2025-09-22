# Navigation and Guidance Data Analysis Utilities
# ===============================================
# Extract and visualize navigation and guidance sensor data from SOLAQUA CSV exports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from datetime import datetime
import json

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


class NavigationGuidanceAnalyzer:
    """
    Main class for analyzing navigation and guidance data from CSV files
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
        """Discover available navigation and guidance files"""
        # Target sensor types
        target_sensors = [
            "guidance",
            "navigation_plane_approximation", 
            "navigation_plane_approximation_position"
        ]
        
        bags = set()
        sensors = set()
        
        # Find all matching files
        for sensor in target_sensors:
            files = list(self.by_bag_folder.glob(f"{sensor}__*.csv"))
            
            for file in files:
                parts = file.stem.split("__")
                if len(parts) >= 2:
                    sensor_name = parts[0]
                    bag_name = parts[1].replace("_data", "")
                    
                    sensors.add(sensor_name)
                    bags.add(bag_name)
        
        self.available_bags = sorted(list(bags))
        self.available_sensors = sorted(list(sensors))
        
        print(f"🔍 Found Navigation/Guidance data:")
        print(f"   📅 Bags: {len(self.available_bags)}")
        print(f"   📊 Sensors: {len(self.available_sensors)}")
        print(f"   Sensors: {', '.join(self.available_sensors)}")
    
    def load_sensor_data(self, sensor_type, bag_name=None, verbose=True):
        """
        Load data for a specific sensor type and bag
        
        Parameters:
        -----------
        sensor_type : str
            Type of sensor (guidance, navigation_plane_approximation, etc.)
        bag_name : str or None
            Specific bag to load, or None for first available
        verbose : bool
            Print loading information
            
        Returns:
        --------
        pandas.DataFrame or None
            Loaded data with time processing
        """
        if bag_name is None:
            bag_name = self.available_bags[0] if self.available_bags else None
            
        if bag_name is None:
            print("❌ No bags available")
            return None
            
        return self._load_single_file(sensor_type, bag_name, verbose)
    
    def _load_single_file(self, sensor_prefix, bag_name, verbose=True):
        """Load a single CSV file with error handling and time processing"""
        file_pattern = f"{sensor_prefix}__{bag_name}_data.csv"
        file_path = self.by_bag_folder / file_pattern
        
        if not file_path.exists():
            if verbose:
                print(f"❌ File not found: {file_pattern}")
            return None
        
        try:
            data = pd.read_csv(file_path)
            
            # Process timestamps
            if 'ts_oslo' in data.columns:
                data['datetime'] = pd.to_datetime(data['ts_oslo'])
            elif 't' in data.columns:
                data['datetime'] = pd.to_datetime(data['t'], unit='s')
            
            # Calculate relative time in minutes
            if 't_rel' in data.columns:
                data['t_rel_min'] = data['t_rel'] / 60.0
            elif 't' in data.columns:
                data['t_rel'] = data['t'] - data['t'].min()
                data['t_rel_min'] = data['t_rel'] / 60.0
            
            if verbose:
                print(f"📁 Loaded {file_pattern}: {len(data)} rows, {len(data.columns)} columns")
            
            return data
            
        except Exception as e:
            if verbose:
                print(f"❌ Error loading {file_pattern}: {e}")
            return None
    
    def analyze_guidance_errors(self, bag_name=None, interactive=True):
        """
        Analyze guidance system error components
        
        Parameters:
        -----------
        bag_name : str or None
            Bag to analyze
        interactive : bool
            Use plotly for interactive plots
        """
        if bag_name is None:
            bag_name = self.available_bags[0]
        
        guidance_data = self.load_sensor_data("guidance", bag_name)
        
        if guidance_data is None:
            print("❌ No guidance data available")
            return
        
        print(f"🎯 Analyzing guidance errors for bag: {bag_name}")
        
        # Error columns
        error_cols = [col for col in guidance_data.columns if col.startswith('error_')]
        desired_cols = [col for col in guidance_data.columns if col.startswith('desired_')]
        
        if interactive and HAS_PLOTLY:
            self._plot_guidance_errors_plotly(guidance_data, error_cols, desired_cols)
        else:
            self._plot_guidance_errors_matplotlib(guidance_data, error_cols, desired_cols)
    
    def _plot_guidance_errors_plotly(self, data, error_cols, desired_cols):
        """Create interactive guidance error plots with plotly"""
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Position Errors (m)', 'Desired vs Error',
                'Attitude Errors (deg)', 'Net Distance Error',
                'Velocity Errors (m/s)', 'Error Statistics'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Position errors
        pos_errors = ['error_x', 'error_y', 'error_z']
        for i, col in enumerate(pos_errors):
            if col in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['t_rel_min'], y=data[col], 
                             name=col.replace('error_', ''), 
                             line=dict(color=px.colors.qualitative.Set1[i])),
                    row=1, col=1
                )
        
        # Desired vs errors
        if 'desired_surge' in data.columns and 'error_surge' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['t_rel_min'], y=data['desired_surge'], 
                         name='Desired Surge', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=data['t_rel_min'], y=data['error_surge'], 
                         name='Error Surge', line=dict(color='red')),
                row=1, col=2
            )
        
        # Attitude errors
        if 'error_yaw' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['t_rel_min'], y=np.degrees(data['error_yaw']), 
                         name='Yaw Error', line=dict(color='green')),
                row=2, col=1
            )
        
        # Net distance error
        if 'error_net_distance' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['t_rel_min'], y=data['error_net_distance'], 
                         name='Net Distance Error', line=dict(color='purple')),
                row=2, col=2
            )
        
        # Velocity errors
        vel_errors = ['error_surge', 'error_sway', 'error_heave']
        for i, col in enumerate(vel_errors):
            if col in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['t_rel_min'], y=data[col], 
                             name=col.replace('error_', ''), 
                             line=dict(color=px.colors.qualitative.Set2[i])),
                    row=3, col=1
                )
        
        # Error statistics as bar chart
        error_stats = {}
        for col in error_cols:
            if col in data.columns:
                error_stats[col.replace('error_', '')] = data[col].abs().mean()
        
        if error_stats:
            fig.add_trace(
                go.Bar(x=list(error_stats.keys()), y=list(error_stats.values()),
                       name='Mean Absolute Error'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Guidance System Error Analysis",
            showlegend=True
        )
        
        # Update x-axis labels
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Time (minutes)", row=i, col=j)
        
        fig.show()
    
    def _plot_guidance_errors_matplotlib(self, data, error_cols, desired_cols):
        """Create guidance error plots with matplotlib"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Guidance System Error Analysis', fontsize=16)
        
        # Position errors
        pos_errors = ['error_x', 'error_y', 'error_z']
        for col in pos_errors:
            if col in data.columns:
                axes[0,0].plot(data['t_rel_min'], data[col], label=col.replace('error_', ''))
        axes[0,0].set_title('Position Errors (m)')
        axes[0,0].set_xlabel('Time (minutes)')
        axes[0,0].set_ylabel('Error (m)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Desired vs actual
        if 'desired_surge' in data.columns and 'error_surge' in data.columns:
            axes[0,1].plot(data['t_rel_min'], data['desired_surge'], label='Desired Surge', color='blue')
            axes[0,1].plot(data['t_rel_min'], data['error_surge'], label='Error Surge', color='red')
        axes[0,1].set_title('Desired vs Error')
        axes[0,1].set_xlabel('Time (minutes)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Attitude errors
        if 'error_yaw' in data.columns:
            axes[1,0].plot(data['t_rel_min'], np.degrees(data['error_yaw']), label='Yaw Error', color='green')
        axes[1,0].set_title('Attitude Errors (degrees)')
        axes[1,0].set_xlabel('Time (minutes)')
        axes[1,0].set_ylabel('Error (degrees)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Net distance error
        if 'error_net_distance' in data.columns:
            axes[1,1].plot(data['t_rel_min'], data['error_net_distance'], label='Net Distance Error', color='purple')
        axes[1,1].set_title('Net Distance Error')
        axes[1,1].set_xlabel('Time (minutes)')
        axes[1,1].set_ylabel('Error')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Velocity errors
        vel_errors = ['error_surge', 'error_sway', 'error_heave']
        for col in vel_errors:
            if col in data.columns:
                axes[2,0].plot(data['t_rel_min'], data[col], label=col.replace('error_', ''))
        axes[2,0].set_title('Velocity Errors (m/s)')
        axes[2,0].set_xlabel('Time (minutes)')
        axes[2,0].set_ylabel('Error (m/s)')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # Error statistics
        error_stats = {}
        for col in error_cols:
            if col in data.columns:
                error_stats[col.replace('error_', '')] = data[col].abs().mean()
        
        if error_stats:
            axes[2,1].bar(error_stats.keys(), error_stats.values())
            axes[2,1].set_title('Mean Absolute Errors')
            axes[2,1].set_xlabel('Error Type')
            axes[2,1].set_ylabel('Mean Absolute Error')
            axes[2,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_navigation_plane(self, bag_name=None, interactive=True):
        """
        Analyze navigation plane approximation data
        
        Parameters:
        -----------
        bag_name : str or None
            Bag to analyze
        interactive : bool
            Use plotly for interactive plots
        """
        if bag_name is None:
            bag_name = self.available_bags[0]
        
        plane_data = self.load_sensor_data("navigation_plane_approximation", bag_name)
        plane_pos_data = self.load_sensor_data("navigation_plane_approximation_position", bag_name)
        
        if plane_data is None:
            print("❌ No navigation plane approximation data available")
            return
        
        print(f"🧭 Analyzing navigation plane for bag: {bag_name}")
        
        if interactive and HAS_PLOTLY:
            self._plot_navigation_plane_plotly(plane_data, plane_pos_data)
        else:
            self._plot_navigation_plane_matplotlib(plane_data, plane_pos_data)
    
    def _plot_navigation_plane_plotly(self, plane_data, plane_pos_data):
        """Create interactive navigation plane plots with plotly"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Net Distance & Altitude', 'Net Velocity Components',
                '3D Position Trajectory', 'Net Heading & Pitch'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"type": "scatter3d"}, {"secondary_y": False}]]
        )
        
        # Net distance and altitude
        if 'NetDistance' in plane_data.columns:
            fig.add_trace(
                go.Scatter(x=plane_data['t_rel_min'], y=plane_data['NetDistance'], 
                         name='Net Distance', line=dict(color='blue')),
                row=1, col=1
            )
        
        if 'Altitude' in plane_data.columns:
            fig.add_trace(
                go.Scatter(x=plane_data['t_rel_min'], y=plane_data['Altitude'], 
                         name='Altitude', line=dict(color='red')),
                row=1, col=1, secondary_y=True
            )
        
        # Net velocity components
        vel_components = ['NetVelocity_u', 'NetVelocity_v', 'NetVelocity_w']
        colors = ['blue', 'green', 'red']
        for i, col in enumerate(vel_components):
            if col in plane_data.columns:
                fig.add_trace(
                    go.Scatter(x=plane_data['t_rel_min'], y=plane_data[col], 
                             name=col, line=dict(color=colors[i])),
                    row=1, col=2
                )
        
        # 3D trajectory if position data available
        if plane_pos_data is not None and all(col in plane_pos_data.columns for col in ['x', 'y', 'z']):
            fig.add_trace(
                go.Scatter3d(x=plane_pos_data['x'], y=plane_pos_data['y'], z=plane_pos_data['z'],
                           mode='lines+markers', name='3D Trajectory',
                           line=dict(color='blue', width=4),
                           marker=dict(size=3)),
                row=2, col=1
            )
        
        # Heading and pitch
        if 'NetHeading' in plane_data.columns:
            fig.add_trace(
                go.Scatter(x=plane_data['t_rel_min'], y=np.degrees(plane_data['NetHeading']), 
                         name='Net Heading (deg)', line=dict(color='purple')),
                row=2, col=2
            )
        
        if 'NetPitch' in plane_data.columns:
            fig.add_trace(
                go.Scatter(x=plane_data['t_rel_min'], y=np.degrees(plane_data['NetPitch']), 
                         name='Net Pitch (deg)', line=dict(color='orange')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Navigation Plane Approximation Analysis",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (minutes)", row=1, col=1)
        fig.update_xaxes(title_text="Time (minutes)", row=1, col=2)
        fig.update_xaxes(title_text="Time (minutes)", row=2, col=2)
        
        fig.show()
    
    def _plot_navigation_plane_matplotlib(self, plane_data, plane_pos_data):
        """Create navigation plane plots with matplotlib"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Navigation Plane Approximation Analysis', fontsize=16)
        
        # Net distance and altitude
        ax1 = axes[0,0]
        if 'NetDistance' in plane_data.columns:
            ax1.plot(plane_data['t_rel_min'], plane_data['NetDistance'], 'b-', label='Net Distance')
            ax1.set_ylabel('Net Distance (m)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
        
        if 'Altitude' in plane_data.columns:
            ax2 = ax1.twinx()
            ax2.plot(plane_data['t_rel_min'], plane_data['Altitude'], 'r-', label='Altitude')
            ax2.set_ylabel('Altitude (m)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title('Net Distance & Altitude')
        ax1.set_xlabel('Time (minutes)')
        ax1.grid(True, alpha=0.3)
        
        # Net velocity components
        vel_components = ['NetVelocity_u', 'NetVelocity_v', 'NetVelocity_w']
        for col in vel_components:
            if col in plane_data.columns:
                axes[0,1].plot(plane_data['t_rel_min'], plane_data[col], label=col)
        axes[0,1].set_title('Net Velocity Components')
        axes[0,1].set_xlabel('Time (minutes)')
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3D trajectory if available
        if plane_pos_data is not None and all(col in plane_pos_data.columns for col in ['x', 'y']):
            axes[1,0].plot(plane_pos_data['x'], plane_pos_data['y'], 'b-', alpha=0.7)
            axes[1,0].scatter(plane_pos_data['x'].iloc[0], plane_pos_data['y'].iloc[0], 
                            color='green', s=100, label='Start', zorder=5)
            axes[1,0].scatter(plane_pos_data['x'].iloc[-1], plane_pos_data['y'].iloc[-1], 
                            color='red', s=100, label='End', zorder=5)
            axes[1,0].set_title('2D Position Trajectory')
            axes[1,0].set_xlabel('X Position (m)')
            axes[1,0].set_ylabel('Y Position (m)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].axis('equal')
        
        # Heading and pitch
        if 'NetHeading' in plane_data.columns:
            axes[1,1].plot(plane_data['t_rel_min'], np.degrees(plane_data['NetHeading']), 
                          label='Net Heading', color='purple')
        
        if 'NetPitch' in plane_data.columns:
            axes[1,1].plot(plane_data['t_rel_min'], np.degrees(plane_data['NetPitch']), 
                          label='Net Pitch', color='orange')
        
        axes[1,1].set_title('Net Heading & Pitch')
        axes[1,1].set_xlabel('Time (minutes)')
        axes[1,1].set_ylabel('Angle (degrees)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_navigation_guidance(self, bag_name=None, interactive=True):
        """
        Compare navigation and guidance systems
        
        Parameters:
        -----------
        bag_name : str or None
            Bag to analyze
        interactive : bool
            Use plotly for interactive plots
        """
        if bag_name is None:
            bag_name = self.available_bags[0]
        
        guidance_data = self.load_sensor_data("guidance", bag_name)
        plane_data = self.load_sensor_data("navigation_plane_approximation", bag_name)
        plane_pos_data = self.load_sensor_data("navigation_plane_approximation_position", bag_name)
        
        if guidance_data is None or plane_data is None:
            print("❌ Missing guidance or navigation data")
            return
        
        print(f"⚖️  Comparing navigation and guidance for bag: {bag_name}")
        
        # Print summary statistics
        print(f"\n📊 Data Summary:")
        print(f"  - Guidance samples: {len(guidance_data)}")
        print(f"  - Navigation plane samples: {len(plane_data)}")
        if plane_pos_data is not None:
            print(f"  - Navigation position samples: {len(plane_pos_data)}")
        
        # Compare sampling rates
        guidance_duration = guidance_data['t_rel'].max() / 60.0
        plane_duration = plane_data['t_rel'].max() / 60.0
        
        print(f"\n🕐 Sampling Rates:")
        print(f"  - Guidance: {len(guidance_data)/guidance_duration:.1f} samples/min")
        print(f"  - Navigation plane: {len(plane_data)/plane_duration:.1f} samples/min")
        
        if interactive and HAS_PLOTLY:
            self._plot_comparison_plotly(guidance_data, plane_data, plane_pos_data)
        else:
            self._plot_comparison_matplotlib(guidance_data, plane_data, plane_pos_data)
    
    def _plot_comparison_plotly(self, guidance_data, plane_data, plane_pos_data):
        """Create comparison plots with plotly"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Position: Guidance vs Navigation', 'Velocity Comparison',
                'Error vs Net Distance', 'Guidance Desired vs Navigation Actual'
            ]
        )
        
        # Position comparison
        if 'error_x' in guidance_data.columns and 'error_y' in guidance_data.columns:
            fig.add_trace(
                go.Scatter(x=guidance_data['error_x'], y=guidance_data['error_y'],
                         mode='lines', name='Guidance Errors',
                         line=dict(color='red')),
                row=1, col=1
            )
        
        if plane_pos_data is not None and 'x' in plane_pos_data.columns:
            # Normalize positions for comparison
            x_norm = plane_pos_data['x'] - plane_pos_data['x'].mean()
            y_norm = plane_pos_data['y'] - plane_pos_data['y'].mean()
            fig.add_trace(
                go.Scatter(x=x_norm, y=y_norm,
                         mode='lines', name='Navigation Position',
                         line=dict(color='blue')),
                row=1, col=1
            )
        
        # Velocity comparison
        if 'NetVelocity_u' in plane_data.columns:
            fig.add_trace(
                go.Scatter(x=plane_data['t_rel_min'], y=plane_data['NetVelocity_u'],
                         name='Nav Velocity U', line=dict(color='blue')),
                row=1, col=2
            )
        
        if 'desired_surge' in guidance_data.columns:
            fig.add_trace(
                go.Scatter(x=guidance_data['t_rel_min'], y=guidance_data['desired_surge'],
                         name='Guidance Desired Surge', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
        
        # Error vs net distance
        if 'error_net_distance' in guidance_data.columns and 'NetDistance' in plane_data.columns:
            fig.add_trace(
                go.Scatter(x=guidance_data['t_rel_min'], y=guidance_data['error_net_distance'],
                         name='Guidance Error', line=dict(color='red')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=plane_data['t_rel_min'], y=plane_data['NetDistance'],
                         name='Navigation Distance', line=dict(color='blue')),
                row=2, col=1
            )
        
        # Desired vs actual comparison
        if 'desired_yaw' in guidance_data.columns and 'NetHeading' in plane_data.columns:
            fig.add_trace(
                go.Scatter(x=guidance_data['t_rel_min'], y=np.degrees(guidance_data['desired_yaw']),
                         name='Desired Yaw', line=dict(color='green', dash='dash')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=plane_data['t_rel_min'], y=np.degrees(plane_data['NetHeading']),
                         name='Actual Heading', line=dict(color='blue')),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Navigation vs Guidance System Comparison",
            showlegend=True
        )
        
        fig.show()
    
    def _plot_comparison_matplotlib(self, guidance_data, plane_data, plane_pos_data):
        """Create comparison plots with matplotlib"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Navigation vs Guidance System Comparison', fontsize=16)
        
        # Position comparison
        if 'error_x' in guidance_data.columns and 'error_y' in guidance_data.columns:
            axes[0,0].plot(guidance_data['error_x'], guidance_data['error_y'], 
                          'r-', label='Guidance Errors', alpha=0.7)
        
        if plane_pos_data is not None and 'x' in plane_pos_data.columns:
            x_norm = plane_pos_data['x'] - plane_pos_data['x'].mean()
            y_norm = plane_pos_data['y'] - plane_pos_data['y'].mean()
            axes[0,0].plot(x_norm, y_norm, 'b-', label='Navigation Position', alpha=0.7)
        
        axes[0,0].set_title('Position: Guidance vs Navigation')
        axes[0,0].set_xlabel('X Component')
        axes[0,0].set_ylabel('Y Component')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axis('equal')
        
        # Velocity comparison
        if 'NetVelocity_u' in plane_data.columns:
            axes[0,1].plot(plane_data['t_rel_min'], plane_data['NetVelocity_u'], 
                          'b-', label='Nav Velocity U')
        
        if 'desired_surge' in guidance_data.columns:
            axes[0,1].plot(guidance_data['t_rel_min'], guidance_data['desired_surge'], 
                          'r--', label='Guidance Desired Surge')
        
        axes[0,1].set_title('Velocity Comparison')
        axes[0,1].set_xlabel('Time (minutes)')
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Error vs net distance
        if 'error_net_distance' in guidance_data.columns:
            axes[1,0].plot(guidance_data['t_rel_min'], guidance_data['error_net_distance'], 
                          'r-', label='Guidance Error')
        
        if 'NetDistance' in plane_data.columns:
            axes[1,0].plot(plane_data['t_rel_min'], plane_data['NetDistance'], 
                          'b-', label='Navigation Distance')
        
        axes[1,0].set_title('Error vs Net Distance')
        axes[1,0].set_xlabel('Time (minutes)')
        axes[1,0].set_ylabel('Distance/Error')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Desired vs actual heading
        if 'desired_yaw' in guidance_data.columns:
            axes[1,1].plot(guidance_data['t_rel_min'], np.degrees(guidance_data['desired_yaw']), 
                          'g--', label='Desired Yaw')
        
        if 'NetHeading' in plane_data.columns:
            axes[1,1].plot(plane_data['t_rel_min'], np.degrees(plane_data['NetHeading']), 
                          'b-', label='Actual Heading')
        
        axes[1,1].set_title('Desired vs Actual Heading')
        axes[1,1].set_xlabel('Time (minutes)')
        axes[1,1].set_ylabel('Angle (degrees)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_summary(self, output_file="navigation_guidance_summary.csv"):
        """
        Export a summary of all available navigation and guidance data to CSV
        
        Parameters:
        -----------
        output_file : str
            Output CSV filename
        """
        summary_data = []
        
        for sensor in self.available_sensors:
            for bag in self.available_bags:
                data = self._load_single_file(sensor, bag, verbose=False)
                
                if data is not None:
                    summary_data.append({
                        'sensor': sensor,
                        'bag': bag,
                        'samples': len(data),
                        'duration_min': data['t_rel'].max() / 60.0 if 't_rel' in data.columns else 0,
                        'start_time': data['datetime'].min() if 'datetime' in data.columns else None,
                        'end_time': data['datetime'].max() if 'datetime' in data.columns else None,
                        'columns': len(data.columns),
                        'sampling_rate': len(data) / (data['t_rel'].max() / 60.0) if 't_rel' in data.columns and data['t_rel'].max() > 0 else 0
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to exports/outputs folder
        output_path = Path("exports/outputs") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_df.to_csv(output_path, index=False)
        print(f"✅ Summary exported to: {output_path}")
        
        return summary_df
