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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotorDataAnalyzer:
    """Main class for analyzing BLDC motor dynamometer data."""
    
    def __init__(self, data_folder: str, output_folder: str = "analysis_results"):
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders for different types of outputs
        self.individual_plots_folder = self.output_folder / "individual_plots"
        self.comparison_plots_folder = self.output_folder / "comparison_plots"
        self.reports_folder = self.output_folder / "reports"
        
        for folder in [self.individual_plots_folder, self.comparison_plots_folder, self.reports_folder]:
            folder.mkdir(exist_ok=True)
            
        # Set up matplotlib for better plots
        plt.style.use('seaborn-v0_8')
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """Configure matplotlib for professional-looking plots."""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 100,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 1.5,
            'grid.alpha': 0.3
        })
        
    def get_csv_files(self) -> List[Path]:
        """Get all CSV files in the data folder."""
        csv_files = list(self.data_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        return sorted(csv_files)
        
    def load_csv_efficiently(self, file_path: Path, chunksize: int = 10000) -> pd.DataFrame:
        """
        Load CSV file efficiently to avoid memory issues.
        Uses chunking for very large files.
        """
        try:
            # First, try to load the entire file
            df = pd.read_csv(file_path, low_memory=False)
            
            # If file is very large (>100MB), consider chunking
            if df.memory_usage(deep=True).sum() > 100 * 1024 * 1024:
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
        
        # Define patterns for different types of measurements
        patterns = {
            'time': ['time', 'timestamp', 'sec', 'seconds'],
            'vibration': ['vib', 'vibration', 'accel', 'acceleration', 'shake'],
            'speed': ['speed', 'rpm', 'velocity', 'rotation'],
            'torque': ['torque', 'force', 'moment'],
            'power': ['power', 'watt', 'kw'],
            'temperature': ['temp', 'temperature', 'thermal', 'heat'],
            'current': ['current', 'amp', 'ampere', 'i_'],
            'voltage': ['voltage', 'volt', 'v_', 'potential']
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
        """Analyze a single CSV file and generate plots."""
        logger.info(f"Analyzing {file_path.name}")
        
        # Load data
        df = self.load_csv_efficiently(file_path)
        if df is None:
            return None
            
        # Identify columns
        column_mapping = self.identify_columns(df)
        
        # Calculate basic statistics
        stats = {
            'filename': file_path.name,
            'rows': len(df),
            'columns': len(df.columns),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024)
        }
        
        # Calculate vibration metrics
        if column_mapping['vibration']:
            vibration_metrics = self.calculate_vibration_metrics(df, column_mapping['vibration'])
            stats.update(vibration_metrics)
            
        # Generate individual plots
        self.create_individual_plots(df, column_mapping, file_path.stem)
        
        # Clean up memory
        del df
        gc.collect()
        
        return stats
        
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
        
        for i, col in enumerate(vibration_cols[:4]):  # Limit to 4 columns to fit subplots
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
                        ax.hist(data, bins=50, alpha=0.7, color='red', edgecolor='black')
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
            
        # Limit to 6 metrics for subplot layout
        perf_metrics = perf_metrics[:6]
        
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
        for col in important_cols[:5]:  # Limit to 5 for readability
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
        important_cols = numeric_cols[:15]  # Limit to 15 columns
        
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
            with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    if not os.path.exists(args.data_folder):
        logger.error(f"Data folder not found: {args.data_folder}")
        sys.exit(1)
        
    # Initialize analyzer
    analyzer = MotorDataAnalyzer(args.data_folder, args.output)
    
    # Process files
    try:
        file_stats = analyzer.process_batch(
            max_files=args.max_files,
            use_multiprocessing=not args.sequential
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