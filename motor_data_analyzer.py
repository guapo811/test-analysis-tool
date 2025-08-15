#!/usr/bin/env python3
"""
BLDC Motor Dynamometer Data Analysis Tool
Analyzes CSV data from motor tests with focus on vibration and performance metrics.
Designed for memory-efficient processing of thousands of files.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Optional, Tuple
import warnings
from datetime import datetime
import gc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json

# Import new modules
from signal_processing import SignalProcessor
from data_quality import DataQualityManager
from predictive_analytics import PredictiveAnalyzer
from motor_analysis import MotorAnalyzer
from error_handling import ErrorHandler, error_handler_decorator, ContextualErrorHandler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotorDataAnalyzer:
    """Main class for analyzing BLDC motor dynamometer data."""
    
    def __init__(self, data_folder: str, output_folder: str = "analysis_results", config_path: str = "config.json"):
        # Load configuration
        self.config = self.load_config(config_path)
        
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder or self.config['output_settings']['output_folder'])
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders for different types of outputs
        self.individual_plots_folder = self.output_folder / "individual_plots"
        self.comparison_plots_folder = self.output_folder / "comparison_plots"
        self.reports_folder = self.output_folder / "reports"
        
        for folder in [self.individual_plots_folder, self.comparison_plots_folder, self.reports_folder]:
            folder.mkdir(exist_ok=True)
            
        # Set up matplotlib for better plots
        plot_style = self.config['plot_settings']['plot_style']
        plt.style.use(plot_style)
        self.setup_plot_style()
        
        # Initialize new analysis modules
        self.signal_processor = SignalProcessor(self.config)
        self.data_quality_manager = DataQualityManager(self.config)
        self.predictive_analyzer = PredictiveAnalyzer(self.config)
        self.motor_analyzer = MotorAnalyzer(self.config)
        self.error_handler = ErrorHandler(self.config)
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_path}: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration if config file is not available."""
        return {
            'output_settings': {'output_folder': 'analysis_results'},
            'plot_settings': {
                'plot_style': 'seaborn-v0_8',
                'figure_size': [12, 8],
                'line_width': 1.5,
                'grid_alpha': 0.3,
                'font_sizes': {'title': 14, 'axis_label': 12, 'tick_label': 10, 'legend': 10}
            },
            'column_detection': {
                'time_keywords': ['time', 'timestamp', 'sec', 'seconds'],
                'vibration_keywords': ['vib', 'vibration', 'accel', 'acceleration', 'shake'],
                'speed_keywords': ['speed', 'rpm', 'velocity', 'rotation'],
                'torque_keywords': ['torque', 'force', 'moment'],
                'power_keywords': ['power', 'watt', 'kw'],
                'temperature_keywords': ['temp', 'temperature', 'thermal', 'heat'],
                'current_keywords': ['current', 'amp', 'ampere', 'i_'],
                'voltage_keywords': ['voltage', 'volt', 'v_', 'potential']
            },
            'data_settings': {'chunk_size': 10000, 'large_file_threshold_mb': 100},
            'processing_settings': {'use_multiprocessing': True, 'max_workers': 4},
            'vibration_analysis': {'max_vibration_columns': 4},
            'individual_plots': {
                'vibration_analysis': {'bins': 50},
                'performance_overview': {'max_metrics': 6},
                'time_series': {'max_series': 5},
                'correlation_matrix': {'max_columns': 15}
            },
            'signal_processing': {
                'enable_fft_analysis': True,
                'fft_window': 'hann',
                'fft_nperseg': 1024,
                'fft_overlap': 0.5,
                'enable_filtering': True,
                'enable_envelope_analysis': True
            },
            'data_quality': {
                'enable_validation': True,
                'enable_cleaning': True,
                'missing_data_threshold': 0.1,
                'interpolation_method': 'linear',
                'outlier_detection': {'enabled': True, 'method': 'iqr', 'threshold': 1.5, 'auto_remove': False}
            },
            'predictive_analytics': {
                'enable_anomaly_detection': True,
                'anomaly_methods': ['isolation_forest'],
                'contamination_rate': 0.1,
                'enable_trend_analysis': True,
                'trend_window_size': 100,
                'enable_pattern_recognition': True,
                'model_training': {'auto_train': False, 'validation_split': 0.2}
            },
            'motor_analysis': {
                'enable_efficiency_calculation': True,
                'enable_torque_ripple_analysis': True,
                'torque_ripple_harmonics': [1, 2, 3, 6, 12],
                'enable_harmonic_analysis': True,
                'harmonic_orders': [1, 2, 3, 5, 7],
                'enable_cogging_detection': True,
                'temperature_correlation': True,
                'load_mapping': {'enabled': True, 'load_ranges': [0, 25, 50, 75, 100]}
            },
            'error_handling': {
                'enable_graceful_recovery': True,
                'detailed_logging': True,
                'auto_repair_attempts': 3,
                'backup_corrupted_files': True,
                'continue_on_error': True,
                'error_report_generation': True
            }
        }
        
    def setup_plot_style(self):
        """Configure matplotlib for professional-looking plots."""
        plot_config = self.config['plot_settings']
        font_sizes = plot_config['font_sizes']
        
        plt.rcParams.update({
            'figure.figsize': tuple(plot_config['figure_size']),
            'figure.dpi': 100,
            'axes.titlesize': font_sizes['title'],
            'axes.labelsize': font_sizes['axis_label'],
            'xtick.labelsize': font_sizes['tick_label'],
            'ytick.labelsize': font_sizes['tick_label'],
            'legend.fontsize': font_sizes['legend'],
            'lines.linewidth': plot_config['line_width'],
            'grid.alpha': plot_config['grid_alpha']
        })
        
    def get_csv_files(self) -> List[Path]:
        """Get all CSV files in the data folder."""
        csv_files = list(self.data_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        return sorted(csv_files)
        
    def load_csv_efficiently(self, file_path: Path, chunksize: int = None) -> pd.DataFrame:
        """
        Load CSV file efficiently to avoid memory issues.
        Uses chunking for very large files.
        """
        if chunksize is None:
            chunksize = self.config['data_settings']['chunk_size']
            
        try:
            # First, try to load the entire file
            df = pd.read_csv(file_path, low_memory=False)
            
            # If file is very large, consider chunking
            threshold_mb = self.config['data_settings']['large_file_threshold_mb']
            if df.memory_usage(deep=True).sum() > threshold_mb * 1024 * 1024:
                logger.warning(f"Large file detected: {file_path.name}. Consider using chunked processing.")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
            
    def identify_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically identify relevant columns based on common naming patterns.
        """
        columns = df.columns.str.lower()
        
        column_mapping = {
            'time': [],
            'vibration': [],
            'speed': [],
            'torque': [],
            'power': [],
            'temperature': [],
            'current': [],
            'voltage': [],
            'other': []
        }
        
        # Define patterns for different types of measurements from config
        col_config = self.config['column_detection']
        patterns = {
            'time': col_config['time_keywords'],
            'vibration': col_config['vibration_keywords'],
            'speed': col_config['speed_keywords'],
            'torque': col_config['torque_keywords'],
            'power': col_config['power_keywords'],
            'temperature': col_config['temperature_keywords'],
            'current': col_config['current_keywords'],
            'voltage': col_config['voltage_keywords']
        }
        
        for col in df.columns:
            col_lower = col.lower()
            categorized = False
            
            for category, keywords in patterns.items():
                if any(keyword in col_lower for keyword in keywords):
                    column_mapping[category].append(col)
                    categorized = True
                    break
                    
            if not categorized:
                column_mapping['other'].append(col)
                
        return column_mapping
        
    def calculate_vibration_metrics(self, df: pd.DataFrame, vibration_cols: List[str]) -> Dict[str, float]:
        """Calculate comprehensive vibration analysis metrics."""
        metrics = {}
        
        for col in vibration_cols:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    # Basic statistics
                    metrics[f'{col}_mean'] = data.mean()
                    metrics[f'{col}_std'] = data.std()
                    metrics[f'{col}_max'] = data.max()
                    metrics[f'{col}_min'] = data.min()
                    metrics[f'{col}_rms'] = np.sqrt((data ** 2).mean())
                    
                    # Peak-to-peak
                    metrics[f'{col}_p2p'] = data.max() - data.min()
                    
                    # Percentiles for outlier analysis
                    metrics[f'{col}_95th'] = data.quantile(0.95)
                    metrics[f'{col}_99th'] = data.quantile(0.99)
                    
                    # Vibration severity (simplified)
                    metrics[f'{col}_severity'] = data.std() / data.mean() if data.mean() != 0 else 0
                    
        return metrics
        
    def analyze_single_file(self, file_path: Path) -> Dict:
        """Analyze a single CSV file and generate comprehensive analysis."""
        logger.info(f"Analyzing {file_path.name}")
        
        with ContextualErrorHandler(self.error_handler, 
                                  {'file_path': str(file_path), 'operation': 'file_analysis'}):
            
            # Validate file integrity first
            file_validation = self.error_handler.validate_file_integrity(file_path)
            if not file_validation['is_valid']:
                logger.error(f"File validation failed for {file_path.name}: {file_validation['issues_found']}")
                self.error_handler.backup_corrupted_file(file_path, "File validation failed")
                return None
            
            # Load data
            df = self.load_csv_efficiently(file_path)
            if df is None:
                return None
                
            # Data quality analysis
            quality_results = self.data_quality_manager.validate_data(df)
            
            # Clean data if enabled
            if quality_results['status'] != 'failed':
                df_cleaned, cleaning_report = self.data_quality_manager.clean_data(df)
            else:
                df_cleaned = df
                cleaning_report = {'status': 'skipped', 'reason': 'validation_failed'}
                
            # Identify columns
            column_mapping = self.identify_columns(df_cleaned)
            
            # Calculate basic statistics
            stats = {
                'filename': file_path.name,
                'rows': len(df_cleaned),
                'columns': len(df_cleaned.columns),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'data_quality': quality_results,
                'cleaning_report': cleaning_report
            }
            
            # Set up signal processor with estimated sampling rate
            if column_mapping['time']:
                time_data = df_cleaned[column_mapping['time'][0]]
                sampling_rate = self.signal_processor.estimate_sampling_rate(time_data)
                self.signal_processor.set_sampling_rate(sampling_rate)
                stats['estimated_sampling_rate'] = sampling_rate
            
            # Advanced signal processing analysis
            signal_analysis_results = self.perform_signal_analysis(df_cleaned, column_mapping)
            stats.update(signal_analysis_results)
            
            # Predictive analytics
            predictive_results = self.perform_predictive_analysis(df_cleaned, column_mapping)
            stats.update(predictive_results)
            
            # Motor-specific analysis
            motor_analysis_results = self.perform_motor_analysis(df_cleaned, column_mapping)
            stats.update(motor_analysis_results)
            
            # Calculate traditional vibration metrics
            if column_mapping['vibration']:
                vibration_metrics = self.calculate_vibration_metrics(df_cleaned, column_mapping['vibration'])
                stats.update(vibration_metrics)
                
            # Generate comprehensive plots
            self.create_comprehensive_plots(df_cleaned, column_mapping, file_path.stem, stats)
            
            # Clean up memory
            del df, df_cleaned
            gc.collect()
            
            return stats
    
    def perform_signal_analysis(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]]) -> Dict:
        """Perform advanced signal processing analysis."""
        signal_results = {}
        
        try:
            # FFT analysis on vibration signals
            for col in column_mapping['vibration'][:2]:  # Limit to first 2 columns
                if col in df.columns:
                    time_col = column_mapping['time'][0] if column_mapping['time'] else None
                    time_data = df[time_col] if time_col else None
                    
                    fft_result = self.signal_processor.compute_fft(df[col], time_data)
                    if fft_result:
                        signal_results[f'{col}_fft'] = fft_result
                        
                    # Envelope analysis
                    envelope_result = self.signal_processor.envelope_analysis(df[col])
                    if envelope_result:
                        signal_results[f'{col}_envelope'] = envelope_result
                        
        except Exception as e:
            logger.error(f"Signal analysis failed: {e}")
            signal_results['signal_analysis_error'] = str(e)
            
        return signal_results
    
    def perform_predictive_analysis(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]]) -> Dict:
        """Perform predictive analytics."""
        predictive_results = {}
        
        try:
            # Anomaly detection
            anomaly_result = self.predictive_analyzer.detect_anomalies(df)
            if anomaly_result:
                predictive_results['anomaly_detection'] = anomaly_result
                
            # Trend analysis on key parameters
            trend_columns = []
            for category in ['vibration', 'speed', 'torque', 'power']:
                trend_columns.extend(column_mapping[category][:1])  # One column per category
                
            if trend_columns:
                trend_result = self.predictive_analyzer.trend_analysis(df, trend_columns)
                if trend_result:
                    predictive_results['trend_analysis'] = trend_result
                    
            # Pattern recognition for vibration data
            if column_mapping['vibration']:
                pattern_result = self.predictive_analyzer.pattern_recognition(df, 'vibration')
                if pattern_result:
                    predictive_results['vibration_patterns'] = pattern_result
                    
        except Exception as e:
            logger.error(f"Predictive analysis failed: {e}")
            predictive_results['predictive_analysis_error'] = str(e)
            
        return predictive_results
    
    def perform_motor_analysis(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]]) -> Dict:
        """Perform motor-specific analysis."""
        motor_results = {}
        
        try:
            # Efficiency calculation
            efficiency_result = self.motor_analyzer.calculate_efficiency(df)
            if efficiency_result:
                motor_results['efficiency_analysis'] = efficiency_result
                
            # Torque ripple analysis
            if column_mapping['torque']:
                ripple_result = self.motor_analyzer.analyze_torque_ripple(df)
                if ripple_result:
                    motor_results['torque_ripple'] = ripple_result
                    
            # Harmonic analysis
            if column_mapping['current']:
                harmonic_result = self.motor_analyzer.analyze_harmonics(df, 'current')
                if harmonic_result:
                    motor_results['current_harmonics'] = harmonic_result
                    
            # Cogging torque detection
            cogging_result = self.motor_analyzer.detect_cogging_torque(df)
            if cogging_result:
                motor_results['cogging_analysis'] = cogging_result
                
            # Temperature correlation
            if column_mapping['temperature']:
                temp_result = self.motor_analyzer.analyze_temperature_correlation(df)
                if temp_result:
                    motor_results['temperature_correlation'] = temp_result
                    
        except Exception as e:
            logger.error(f"Motor analysis failed: {e}")
            motor_results['motor_analysis_error'] = str(e)
            
        return motor_results
    
    def create_comprehensive_plots(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]], 
                                 filename: str, analysis_results: Dict):
        """Create comprehensive plots including new analysis results."""
        # Create original plots
        self.create_individual_plots(df, column_mapping, filename)
        
        # Create advanced signal processing plots
        self.create_signal_processing_plots(df, column_mapping, filename, analysis_results)
        
        # Create motor analysis plots
        self.create_motor_analysis_plots(df, column_mapping, filename, analysis_results)
        
        # Create predictive analytics plots
        self.create_predictive_plots(df, column_mapping, filename, analysis_results)
    
    def create_signal_processing_plots(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]], 
                                     filename: str, analysis_results: Dict):
        """Create signal processing plots."""
        try:
            for col in column_mapping['vibration'][:2]:
                # FFT plots
                fft_key = f'{col}_fft'
                if fft_key in analysis_results:
                    fig = self.signal_processor.plot_fft_analysis(
                        analysis_results[fft_key], 
                        f'FFT Analysis - {col} - {filename}'
                    )
                    if fig:
                        fig.savefig(self.individual_plots_folder / f'{filename}_{col}_fft.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                # Envelope plots
                envelope_key = f'{col}_envelope'
                if envelope_key in analysis_results:
                    fig = self.signal_processor.plot_envelope_analysis(
                        analysis_results[envelope_key], 
                        df[col],
                        f'Envelope Analysis - {col} - {filename}'
                    )
                    if fig:
                        fig.savefig(self.individual_plots_folder / f'{filename}_{col}_envelope.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
        except Exception as e:
            logger.error(f"Signal processing plots failed: {e}")
    
    def create_motor_analysis_plots(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]], 
                                  filename: str, analysis_results: Dict):
        """Create motor-specific analysis plots."""
        try:
            # Efficiency plots
            if 'efficiency_analysis' in analysis_results:
                efficiency_data = analysis_results['efficiency_analysis']
                if 'efficiency_data' in efficiency_data:
                    self._plot_efficiency_analysis(efficiency_data, filename)
                    
            # Torque ripple plots
            if 'torque_ripple' in analysis_results:
                ripple_data = analysis_results['torque_ripple']
                if 'harmonic_analysis' in ripple_data:
                    self._plot_torque_ripple_analysis(ripple_data, filename)
                    
        except Exception as e:
            logger.error(f"Motor analysis plots failed: {e}")
    
    def create_predictive_plots(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]], 
                              filename: str, analysis_results: Dict):
        """Create predictive analytics plots."""
        try:
            # Anomaly detection plots
            if 'anomaly_detection' in analysis_results:
                anomaly_data = analysis_results['anomaly_detection']
                if anomaly_data.get('status') == 'completed':
                    self._plot_anomaly_detection(df, anomaly_data, filename)
                    
        except Exception as e:
            logger.error(f"Predictive plots failed: {e}")
    
    def _plot_efficiency_analysis(self, efficiency_data: Dict, filename: str):
        """Plot efficiency analysis results."""
        if 'efficiency_data' not in efficiency_data:
            return
            
        data = efficiency_data['efficiency_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Motor Efficiency Analysis - {filename}', fontsize=16, fontweight='bold')
        
        # Efficiency vs time
        axes[0, 0].plot(data['efficiency'])
        axes[0, 0].set_title('Efficiency vs Time')
        axes[0, 0].set_ylabel('Efficiency (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power comparison
        axes[0, 1].plot(data['electrical_power'], label='Electrical Power')
        axes[0, 1].plot(data['mechanical_power'], label='Mechanical Power')
        axes[0, 1].set_title('Power Comparison')
        axes[0, 1].set_ylabel('Power (W)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Efficiency histogram
        valid_eff = [e for e in data['efficiency'] if e > 0]
        if valid_eff:
            axes[1, 0].hist(valid_eff, bins=30, alpha=0.7)
            axes[1, 0].set_title('Efficiency Distribution')
            axes[1, 0].set_xlabel('Efficiency (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics
        if 'efficiency_statistics' in efficiency_data:
            stats = efficiency_data['efficiency_statistics']
            axes[1, 1].text(0.1, 0.8, f"Mean: {stats['mean_efficiency']:.2f}%", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f"Max: {stats['max_efficiency']:.2f}%", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f"Min: {stats['min_efficiency']:.2f}%", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.2, f"Std: {stats['std_efficiency']:.2f}%", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Efficiency Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.individual_plots_folder / f'{filename}_efficiency_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_torque_ripple_analysis(self, ripple_data: Dict, filename: str):
        """Plot torque ripple analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Torque Ripple Analysis - {filename}', fontsize=16, fontweight='bold')
        
        # Display ripple statistics
        if 'ripple_statistics' in ripple_data:
            stats = ripple_data['ripple_statistics']
            axes[0, 0].text(0.1, 0.8, f"RMS Ripple: {stats['rms_ripple']:.3f} Nm", transform=axes[0, 0].transAxes)
            axes[0, 0].text(0.1, 0.6, f"P2P Ripple: {stats['peak_to_peak_ripple']:.3f} Nm", transform=axes[0, 0].transAxes)
            axes[0, 0].text(0.1, 0.4, f"Ripple Factor: {stats['ripple_factor']:.2f}%", transform=axes[0, 0].transAxes)
            axes[0, 0].text(0.1, 0.2, f"Mean Torque: {stats['mean_torque']:.3f} Nm", transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Ripple Statistics')
        axes[0, 0].axis('off')
        
        # Hide other subplots for now
        for i in range(1, 4):
            axes.flat[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.individual_plots_folder / f'{filename}_torque_ripple.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_detection(self, df: pd.DataFrame, anomaly_data: Dict, filename: str):
        """Plot anomaly detection results."""
        if 'summary' not in anomaly_data:
            return
            
        summary = anomaly_data['summary']
        anomaly_indices = summary.get('anomaly_indices', [])
        
        if not anomaly_indices:
            return
            
        # Plot first numeric column with anomalies highlighted
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        col = numeric_cols[0]
        data = df[col].dropna()
        
        # Plot normal data
        normal_mask = ~data.index.isin(anomaly_indices)
        ax.scatter(data.index[normal_mask], data.values[normal_mask], 
                  alpha=0.6, c='blue', label='Normal', s=10)
        
        # Plot anomalies
        anomaly_mask = data.index.isin(anomaly_indices)
        if anomaly_mask.any():
            ax.scatter(data.index[anomaly_mask], data.values[anomaly_mask], 
                      alpha=0.8, c='red', label='Anomaly', s=20)
        
        ax.set_title(f'Anomaly Detection - {col} - {filename}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.individual_plots_folder / f'{filename}_anomaly_detection.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_individual_plots(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]], filename: str):
        """Create comprehensive plots for individual file analysis."""
        
        # 1. Vibration Analysis (most important)
        if column_mapping['vibration']:
            self.plot_vibration_analysis(df, column_mapping['vibration'], filename)
            
        # 2. Performance Overview
        self.plot_performance_overview(df, column_mapping, filename)
        
        # 3. Time series plots
        if column_mapping['time']:
            self.plot_time_series(df, column_mapping, filename)
            
        # 4. Correlation matrix
        self.plot_correlation_matrix(df, filename)
        
    def plot_vibration_analysis(self, df: pd.DataFrame, vibration_cols: List[str], filename: str):
        """Create detailed vibration analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Vibration Analysis - {filename}', fontsize=16, fontweight='bold')
        
        max_vib_cols = self.config['vibration_analysis']['max_vibration_columns']
        for i, col in enumerate(vibration_cols[:max_vib_cols]):  # Limit columns based on config
            if col in df.columns:
                data = df[col].dropna()
                
                if len(data) > 0:
                    row, col_idx = divmod(i, 2)
                    ax = axes[row, col_idx]
                    
                    # Time series plot
                    ax.plot(data.index, data.values, alpha=0.7, color='red')
                    ax.set_title(f'{col} - Time Series')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Amplitude')
                    ax.grid(True, alpha=0.3)
                    
                    # Add RMS line
                    rms_value = np.sqrt((data ** 2).mean())
                    ax.axhline(y=rms_value, color='blue', linestyle='--', 
                             label=f'RMS: {rms_value:.3f}')
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.individual_plots_folder / f'{filename}_vibration_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Vibration histogram and distribution
        if vibration_cols:
            fig, axes = plt.subplots(1, min(2, len(vibration_cols)), figsize=(12, 5))
            if len(vibration_cols) == 1:
                axes = [axes]
                
            fig.suptitle(f'Vibration Distribution - {filename}', fontsize=14, fontweight='bold')
            
            for i, col in enumerate(vibration_cols[:2]):
                if col in df.columns:
                    data = df[col].dropna()
                    if len(data) > 0:
                        ax = axes[i] if len(vibration_cols) > 1 else axes[0]
                        bins = self.config['individual_plots']['vibration_analysis']['bins']
                        ax.hist(data, bins=bins, alpha=0.7, color='red', edgecolor='black')
                        ax.set_title(f'{col} Distribution')
                        ax.set_xlabel('Amplitude')
                        ax.set_ylabel('Frequency')
                        ax.grid(True, alpha=0.3)
                        
                        # Add statistics text
                        stats_text = f'Mean: {data.mean():.3f}\nStd: {data.std():.3f}\nRMS: {np.sqrt((data**2).mean()):.3f}'
                        ax.text(0.7, 0.8, stats_text, transform=ax.transAxes, 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(self.individual_plots_folder / f'{filename}_vibration_distribution.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_performance_overview(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]], filename: str):
        """Create performance overview plots."""
        # Count available performance metrics
        perf_metrics = []
        for category in ['speed', 'torque', 'power', 'temperature', 'current', 'voltage']:
            perf_metrics.extend(column_mapping[category])
            
        if len(perf_metrics) == 0:
            return
            
        # Limit metrics based on config
        max_metrics = self.config['individual_plots']['performance_overview']['max_metrics']
        perf_metrics = perf_metrics[:max_metrics]
        
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        fig.suptitle(f'Performance Overview - {filename}', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(perf_metrics):
            if metric in df.columns:
                data = df[metric].dropna()
                if len(data) > 0:
                    axes[i].plot(data.index, data.values, alpha=0.8)
                    axes[i].set_title(f'{metric}')
                    axes[i].set_xlabel('Sample Index')
                    axes[i].set_ylabel('Value')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_val = data.mean()
                    axes[i].axhline(y=mean_val, color='red', linestyle='--', 
                                  label=f'Mean: {mean_val:.2f}')
                    axes[i].legend()
                    
        # Hide unused subplots
        for i in range(len(perf_metrics), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(self.individual_plots_folder / f'{filename}_performance_overview.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_time_series(self, df: pd.DataFrame, column_mapping: Dict[str, List[str]], filename: str):
        """Create time-based analysis if time column exists."""
        time_cols = column_mapping['time']
        if not time_cols:
            return
            
        time_col = time_cols[0]  # Use first time column
        
        # Get important metrics for time series
        important_cols = []
        for category in ['vibration', 'speed', 'torque', 'power']:
            important_cols.extend(column_mapping[category][:2])  # Limit to 2 per category
            
        if len(important_cols) == 0:
            return
            
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Normalize data for comparison
        max_series = self.config['individual_plots']['time_series']['max_series']
        for col in important_cols[:max_series]:  # Limit based on config
            if col in df.columns:
                data = df[col].dropna()
                time_data = df[time_col].iloc[:len(data)].dropna()
                
                if len(data) > 0 and len(time_data) > 0:
                    # Normalize to 0-1 range for comparison
                    normalized_data = (data - data.min()) / (data.max() - data.min())
                    ax.plot(time_data, normalized_data, label=col, alpha=0.8)
                    
        ax.set_title(f'Time Series Analysis - {filename}', fontsize=14, fontweight='bold')
        ax.set_xlabel(time_col)
        ax.set_ylabel('Normalized Value (0-1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.individual_plots_folder / f'{filename}_time_series.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self, df: pd.DataFrame, filename: str):
        """Create correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return
            
        # Limit to most important columns to keep plot readable
        max_cols = self.config['individual_plots']['correlation_matrix']['max_columns']
        important_cols = numeric_cols[:max_cols]  # Limit based on config
        
        correlation_matrix = df[important_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'Correlation Matrix - {filename}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.individual_plots_folder / f'{filename}_correlation.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def compare_files(self, file_stats: List[Dict]):
        """Compare multiple files to identify trends and changes."""
        if len(file_stats) < 2:
            logger.warning("Need at least 2 files for comparison")
            return
            
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(file_stats)
        df_comparison = df_comparison.sort_values('filename')
        
        # Focus on vibration metrics for comparison
        vibration_columns = [col for col in df_comparison.columns if 'vib' in col.lower()]
        
        if vibration_columns:
            self.plot_vibration_trends(df_comparison, vibration_columns)
            
        # Plot file statistics trends
        self.plot_file_statistics(df_comparison)
        
        # Generate comparison report
        self.generate_comparison_report(df_comparison)
        
    def plot_vibration_trends(self, df: pd.DataFrame, vibration_cols: List[str]):
        """Plot vibration trends across files."""
        if len(vibration_cols) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Vibration Trends Across Files', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        plot_types = ['_mean', '_std', '_max', '_rms']
        plot_titles = ['Mean Vibration', 'Vibration Standard Deviation', 'Max Vibration', 'RMS Vibration']
        
        for i, suffix in enumerate(plot_types):
            matching_cols = [col for col in vibration_cols if col.endswith(suffix)]
            
            if matching_cols and i < len(axes):
                for col in matching_cols:
                    if col in df.columns:
                        axes[i].plot(range(len(df)), df[col], marker='o', label=col.replace(suffix, ''))
                        
                axes[i].set_title(plot_titles[i])
                axes[i].set_xlabel('File Index (Time Order)')
                axes[i].set_ylabel('Vibration Amplitude')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig(self.comparison_plots_folder / 'vibration_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_file_statistics(self, df: pd.DataFrame):
        """Plot general file statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('File Statistics Over Time', fontsize=16, fontweight='bold')
        
        # File size trend
        if 'file_size_mb' in df.columns:
            axes[0].plot(range(len(df)), df['file_size_mb'], marker='o', color='blue')
            axes[0].set_title('File Size Trend')
            axes[0].set_xlabel('File Index')
            axes[0].set_ylabel('File Size (MB)')
            axes[0].grid(True, alpha=0.3)
            
        # Number of rows trend
        if 'rows' in df.columns:
            axes[1].plot(range(len(df)), df['rows'], marker='s', color='green')
            axes[1].set_title('Data Points per File')
            axes[1].set_xlabel('File Index')
            axes[1].set_ylabel('Number of Rows')
            axes[1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.comparison_plots_folder / 'file_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_comparison_report(self, df: pd.DataFrame):
        """Generate a comprehensive comparison report."""
        report_path = self.reports_folder / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("BLDC Motor Dynamometer Data Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Files Analyzed: {len(df)}\n\n")
            
            # File summary
            f.write("File Summary:\n")
            f.write("-" * 20 + "\n")
            for _, row in df.iterrows():
                f.write(f"File: {row['filename']}\n")
                f.write(f"  - Size: {row['file_size_mb']:.2f} MB\n")
                f.write(f"  - Data Points: {row['rows']:,}\n")
                f.write(f"  - Columns: {row['columns']}\n")
                
                # Vibration summary if available
                vib_cols = [col for col in row.index if 'vib' in col.lower() and 'mean' in col]
                if vib_cols:
                    f.write(f"  - Vibration Metrics:\n")
                    for col in vib_cols:
                        f.write(f"    * {col}: {row[col]:.4f}\n")
                f.write("\n")
                
            # Overall trends
            f.write("Overall Trends:\n")
            f.write("-" * 20 + "\n")
            
            vibration_mean_cols = [col for col in df.columns if 'vib' in col.lower() and 'mean' in col]
            if vibration_mean_cols:
                for col in vibration_mean_cols:
                    trend = "increasing" if df[col].iloc[-1] > df[col].iloc[0] else "decreasing"
                    change_pct = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]) * 100
                    f.write(f"{col}: {trend} by {abs(change_pct):.1f}%\n")
                    
        logger.info(f"Report saved to {report_path}")
        
    def process_batch(self, max_files: Optional[int] = None, use_multiprocessing: bool = True):
        """Process all CSV files in batch with memory efficiency."""
        csv_files = self.get_csv_files()
        
        if max_files:
            csv_files = csv_files[:max_files]
            
        logger.info(f"Processing {len(csv_files)} files")
        
        file_stats = []
        
        if use_multiprocessing and len(csv_files) > 1:
            # Use multiprocessing for large batches
            max_workers = min(self.config['processing_settings']['max_workers'], mp.cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._process_single_file_wrapper, csv_files))
                file_stats = [r for r in results if r is not None]
        else:
            # Sequential processing
            for file_path in csv_files:
                result = self.analyze_single_file(file_path)
                if result:
                    file_stats.append(result)
                    
        # Generate comparison analysis
        if len(file_stats) > 1:
            self.compare_files(file_stats)
            
        logger.info(f"Analysis complete. Results saved to {self.output_folder}")
        return file_stats
        
    def _process_single_file_wrapper(self, file_path):
        """Wrapper for multiprocessing compatibility."""
        return self.analyze_single_file(file_path)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Analyze BLDC motor dynamometer CSV data')
    parser.add_argument('data_folder', help='Folder containing CSV files')
    parser.add_argument('--output', '-o', default='analysis_results', help='Output folder')
    parser.add_argument('--max-files', '-m', type=int, help='Maximum number of files to process')
    parser.add_argument('--sequential', action='store_true', help='Disable multiprocessing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--config', '-c', default='config.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    if not os.path.exists(args.data_folder):
        logger.error(f"Data folder not found: {args.data_folder}")
        sys.exit(1)
        
    # Initialize analyzer
    analyzer = MotorDataAnalyzer(args.data_folder, args.output, args.config)
    
    # Process files
    try:
        # Get multiprocessing setting from config if not overridden by command line
        use_mp = analyzer.config['processing_settings']['use_multiprocessing'] if not args.sequential else False
        
        file_stats = analyzer.process_batch(
            max_files=args.max_files,
            use_multiprocessing=use_mp
        )
        
        print(f"\nAnalysis Complete!")
        print(f"Processed {len(file_stats)} files")
        print(f"Results saved to: {analyzer.output_folder}")
        print(f"Individual plots: {analyzer.individual_plots_folder}")
        print(f"Comparison plots: {analyzer.comparison_plots_folder}")
        print(f"Reports: {analyzer.reports_folder}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()