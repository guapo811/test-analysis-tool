"""
Data Quality Module for Motor Data Analysis
Provides data validation, cleaning, interpolation, and integrity checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.interpolate import interp1d
import warnings

logger = logging.getLogger(__name__)

class DataQualityManager:
    """Comprehensive data quality management for motor analysis data."""
    
    def __init__(self, config: Dict):
        self.config = config['data_quality']
        self.validation_results = {}
        self.cleaning_report = {}
        
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data validation.
        
        Returns:
            Dictionary containing validation results and recommendations
        """
        if not self.config['enable_validation']:
            return {'status': 'validation_disabled'}
            
        validation_results = {
            'status': 'validated',
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'data_summary': {}
        }
        
        try:
            # Basic data structure validation
            validation_results['data_summary'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'data_types': df.dtypes.to_dict()
            }
            
            # Check for empty dataset
            if df.empty:
                validation_results['errors'].append("Dataset is empty")
                validation_results['status'] = 'failed'
                return validation_results
                
            # Check for missing data
            missing_data_analysis = self._analyze_missing_data(df)
            validation_results.update(missing_data_analysis)
            
            # Check for data type consistency
            type_analysis = self._analyze_data_types(df)
            validation_results.update(type_analysis)
            
            # Check for outliers
            outlier_analysis = self._analyze_outliers(df)
            validation_results.update(outlier_analysis)
            
            # Check for time series integrity
            time_analysis = self._analyze_time_integrity(df)
            validation_results.update(time_analysis)
            
            # Check for duplicate data
            duplicate_analysis = self._analyze_duplicates(df)
            validation_results.update(duplicate_analysis)
            
            # Statistical validation
            stats_analysis = self._analyze_statistical_properties(df)
            validation_results.update(stats_analysis)
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_recommendations(validation_results)
            
            self.validation_results = validation_results
            logger.info(f"Data validation completed. Status: {validation_results['status']}")
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            validation_results['errors'].append(f"Validation process failed: {str(e)}")
            validation_results['status'] = 'failed'
            
        return validation_results
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """Analyze missing data patterns."""
        missing_info = {
            'missing_data': {},
            'missing_patterns': {}
        }
        
        # Calculate missing data per column
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_info['missing_data'] = {
            'total_missing_cells': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
        
        # Check if missing data exceeds threshold
        threshold = self.config['missing_data_threshold'] * 100
        problematic_columns = missing_percentages[missing_percentages > threshold]
        
        if len(problematic_columns) > 0:
            missing_info['warnings'] = [
                f"Columns with >={threshold}% missing data: {list(problematic_columns.index)}"
            ]
            
        # Analyze missing data patterns
        if missing_counts.sum() > 0:
            # Check for systematic missing patterns
            missing_pattern = df.isnull()
            pattern_counts = missing_pattern.value_counts()
            missing_info['missing_patterns'] = {
                'unique_patterns': len(pattern_counts),
                'most_common_pattern_frequency': pattern_counts.iloc[0] if len(pattern_counts) > 0 else 0
            }
            
        return missing_info
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict:
        """Analyze data type consistency and appropriateness."""
        type_info = {
            'data_type_analysis': {},
            'type_recommendations': []
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        text_columns = df.select_dtypes(include=['object']).columns
        datetime_columns = df.select_dtypes(include=['datetime']).columns
        
        type_info['data_type_analysis'] = {
            'numeric_columns': len(numeric_columns),
            'text_columns': len(text_columns),
            'datetime_columns': len(datetime_columns)
        }
        
        # Check for potential numeric columns stored as text
        for col in text_columns:
            try:
                # Try to convert to numeric
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                if numeric_conversion.notna().sum() / len(df) > 0.8:  # 80% convertible
                    type_info['type_recommendations'].append(
                        f"Column '{col}' appears to be numeric but stored as text"
                    )
            except:
                pass
                
        # Check for potential datetime columns
        for col in text_columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
                type_info['type_recommendations'].append(
                    f"Column '{col}' might be a datetime column"
                )
                
        return type_info
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict:
        """Analyze outliers in numeric data."""
        outlier_info = {
            'outlier_analysis': {},
            'outlier_columns': {}
        }
        
        if not self.config['outlier_detection']['enabled']:
            return outlier_info
            
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        method = self.config['outlier_detection']['method']
        threshold = self.config['outlier_detection']['threshold']
        
        total_outliers = 0
        
        for col in numeric_columns:
            data = df[col].dropna()
            if len(data) < 4:
                continue
                
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > threshold]
                
            elif method == 'modified_zscore':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                outliers = data[np.abs(modified_z_scores) > threshold]
                
            else:
                continue
                
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(data)) * 100
                outlier_info['outlier_columns'][col] = {
                    'count': len(outliers),
                    'percentage': outlier_percentage,
                    'values': outliers.tolist()[:10]  # Limit to first 10
                }
                total_outliers += len(outliers)
                
        outlier_info['outlier_analysis'] = {
            'total_outliers': total_outliers,
            'affected_columns': len(outlier_info['outlier_columns']),
            'method_used': method
        }
        
        return outlier_info
    
    def _analyze_time_integrity(self, df: pd.DataFrame) -> Dict:
        """Analyze time series data integrity."""
        time_info = {
            'time_analysis': {},
            'time_warnings': []
        }
        
        # Look for time columns
        time_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'timestamp', 'date']):
                time_columns.append(col)
                
        if not time_columns:
            return time_info
            
        for col in time_columns:
            try:
                # Try to convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    time_data = pd.to_datetime(df[col], errors='coerce')
                else:
                    time_data = df[col]
                    
                time_data_clean = time_data.dropna()
                
                if len(time_data_clean) < 2:
                    continue
                    
                # Check for monotonicity
                is_monotonic = time_data_clean.is_monotonic_increasing
                
                # Calculate time intervals
                time_diffs = time_data_clean.diff().dropna()
                
                # Statistical analysis of time intervals
                time_stats = {
                    'is_monotonic': is_monotonic,
                    'min_interval': time_diffs.min(),
                    'max_interval': time_diffs.max(),
                    'mean_interval': time_diffs.mean(),
                    'std_interval': time_diffs.std(),
                    'zero_intervals': (time_diffs == pd.Timedelta(0)).sum()
                }
                
                time_info['time_analysis'][col] = time_stats
                
                # Generate warnings
                if not is_monotonic:
                    time_info['time_warnings'].append(f"Time column '{col}' is not monotonic")
                    
                if time_stats['zero_intervals'] > 0:
                    time_info['time_warnings'].append(
                        f"Time column '{col}' has {time_stats['zero_intervals']} zero intervals"
                    )
                    
            except Exception as e:
                time_info['time_warnings'].append(f"Could not analyze time column '{col}': {str(e)}")
                
        return time_info
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate data."""
        duplicate_info = {
            'duplicate_analysis': {},
            'duplicate_warnings': []
        }
        
        # Check for complete duplicate rows
        duplicate_rows = df.duplicated().sum()
        
        # Check for duplicate indices
        duplicate_indices = df.index.duplicated().sum()
        
        duplicate_info['duplicate_analysis'] = {
            'duplicate_rows': duplicate_rows,
            'duplicate_indices': duplicate_indices,
            'unique_rows': len(df) - duplicate_rows
        }
        
        if duplicate_rows > 0:
            duplicate_info['duplicate_warnings'].append(
                f"Found {duplicate_rows} duplicate rows"
            )
            
        if duplicate_indices > 0:
            duplicate_info['duplicate_warnings'].append(
                f"Found {duplicate_indices} duplicate indices"
            )
            
        return duplicate_info
    
    def _analyze_statistical_properties(self, df: pd.DataFrame) -> Dict:
        """Analyze statistical properties of the data."""
        stats_info = {
            'statistical_analysis': {},
            'statistical_warnings': []
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            data = df[col].dropna()
            if len(data) < 4:
                continue
                
            try:
                # Basic statistics
                col_stats = {
                    'count': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'range': data.max() - data.min()
                }
                
                # Check for constant data
                if col_stats['std'] == 0:
                    stats_info['statistical_warnings'].append(f"Column '{col}' has constant values")
                    
                # Check for extreme skewness
                if abs(col_stats['skewness']) > 3:
                    stats_info['statistical_warnings'].append(
                        f"Column '{col}' has extreme skewness: {col_stats['skewness']:.2f}"
                    )
                    
                # Check for extreme kurtosis
                if abs(col_stats['kurtosis']) > 7:
                    stats_info['statistical_warnings'].append(
                        f"Column '{col}' has extreme kurtosis: {col_stats['kurtosis']:.2f}"
                    )
                    
                stats_info['statistical_analysis'][col] = col_stats
                
            except Exception as e:
                stats_info['statistical_warnings'].append(
                    f"Could not analyze statistics for column '{col}': {str(e)}"
                )
                
        return stats_info
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate data quality recommendations based on validation results."""
        recommendations = []
        
        # Missing data recommendations
        if 'missing_data' in validation_results:
            missing_data = validation_results['missing_data']
            if missing_data.get('total_missing_cells', 0) > 0:
                recommendations.append("Consider using interpolation for missing data")
                
        # Outlier recommendations
        if 'outlier_analysis' in validation_results:
            outlier_data = validation_results['outlier_analysis']
            if outlier_data.get('total_outliers', 0) > 0:
                recommendations.append("Review and possibly remove or transform outliers")
                
        # Time series recommendations
        if 'time_warnings' in validation_results and validation_results['time_warnings']:
            recommendations.append("Address time series integrity issues")
            
        # Data type recommendations
        if 'type_recommendations' in validation_results:
            recommendations.extend(validation_results['type_recommendations'])
            
        # Duplicate data recommendations
        if 'duplicate_analysis' in validation_results:
            dup_data = validation_results['duplicate_analysis']
            if dup_data.get('duplicate_rows', 0) > 0:
                recommendations.append("Remove duplicate rows")
                
        return recommendations
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean the data based on configuration and validation results.
        
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        if not self.config['enable_cleaning']:
            return df, {'status': 'cleaning_disabled'}
            
        cleaning_report = {
            'status': 'completed',
            'actions_taken': [],
            'rows_before': len(df),
            'rows_after': 0,
            'columns_modified': []
        }
        
        try:
            cleaned_df = df.copy()
            
            # Handle missing data
            if self.config['interpolation_method'] != 'none':
                cleaned_df, interpolation_report = self._handle_missing_data(cleaned_df)
                cleaning_report['actions_taken'].extend(interpolation_report)
                
            # Handle outliers
            if self.config['outlier_detection']['enabled'] and \
               self.config['outlier_detection']['auto_remove']:
                cleaned_df, outlier_report = self._handle_outliers(cleaned_df)
                cleaning_report['actions_taken'].extend(outlier_report)
                
            # Remove duplicates
            if len(cleaned_df) != len(cleaned_df.drop_duplicates()):
                original_len = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed_count = original_len - len(cleaned_df)
                cleaning_report['actions_taken'].append(f"Removed {removed_count} duplicate rows")
                
            # Data type conversions
            cleaned_df, type_report = self._fix_data_types(cleaned_df)
            cleaning_report['actions_taken'].extend(type_report)
            
            cleaning_report['rows_after'] = len(cleaned_df)
            self.cleaning_report = cleaning_report
            
            logger.info(f"Data cleaning completed. Actions: {len(cleaning_report['actions_taken'])}")
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            cleaning_report['status'] = 'failed'
            cleaning_report['error'] = str(e)
            return df, cleaning_report
            
        return cleaned_df, cleaning_report
    
    def _handle_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing data using configured interpolation method."""
        actions = []
        method = self.config['interpolation_method']
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            try:
                if method == 'linear':
                    df[col] = df[col].interpolate(method='linear')
                elif method == 'polynomial':
                    df[col] = df[col].interpolate(method='polynomial', order=2)
                elif method == 'spline':
                    df[col] = df[col].interpolate(method='spline', order=3)
                elif method == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill')
                elif method == 'backward_fill':
                    df[col] = df[col].fillna(method='bfill')
                elif method == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif method == 'median':
                    df[col] = df[col].fillna(df[col].median())
                    
                filled_count = missing_count - df[col].isnull().sum()
                if filled_count > 0:
                    actions.append(f"Interpolated {filled_count} missing values in '{col}' using {method}")
                    
            except Exception as e:
                actions.append(f"Failed to interpolate '{col}': {str(e)}")
                
        return df, actions
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove or transform outliers based on configuration."""
        actions = []
        
        # Re-run outlier detection
        outlier_info = self._analyze_outliers(df)
        
        for col, outlier_data in outlier_info.get('outlier_columns', {}).items():
            try:
                outlier_indices = df[col].isin(outlier_data['values'])
                removed_count = outlier_indices.sum()
                
                # Remove outlier rows
                df = df[~outlier_indices]
                actions.append(f"Removed {removed_count} outlier rows from column '{col}'")
                
            except Exception as e:
                actions.append(f"Failed to remove outliers from '{col}': {str(e)}")
                
        return df, actions
    
    def _fix_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fix data type issues."""
        actions = []
        
        # Convert obvious numeric columns stored as text
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Attempt numeric conversion
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # If most values convert successfully, replace the column
                if numeric_series.notna().sum() / len(df) > 0.8:
                    df[col] = numeric_series
                    actions.append(f"Converted column '{col}' from text to numeric")
                    
            except Exception as e:
                pass
                
        return df, actions
    
    def generate_quality_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive data quality report."""
        if not hasattr(self, 'validation_results'):
            self.validate_data(df)
            
        report_lines = [
            "DATA QUALITY REPORT",
            "=" * 50,
            "",
            f"Dataset Overview:",
            f"  - Rows: {self.validation_results['data_summary']['rows']:,}",
            f"  - Columns: {self.validation_results['data_summary']['columns']}",
            f"  - Memory Usage: {self.validation_results['data_summary']['memory_usage_mb']:.2f} MB",
            "",
            f"Validation Status: {self.validation_results['status'].upper()}",
            ""
        ]
        
        # Add warnings
        if self.validation_results.get('warnings'):
            report_lines.extend([
                "WARNINGS:",
                "-" * 20
            ])
            for warning in self.validation_results['warnings']:
                report_lines.append(f"  - {warning}")
            report_lines.append("")
            
        # Add errors
        if self.validation_results.get('errors'):
            report_lines.extend([
                "ERRORS:",
                "-" * 20
            ])
            for error in self.validation_results['errors']:
                report_lines.append(f"  - {error}")
            report_lines.append("")
            
        # Add recommendations
        if self.validation_results.get('recommendations'):
            report_lines.extend([
                "RECOMMENDATIONS:",
                "-" * 20
            ])
            for rec in self.validation_results['recommendations']:
                report_lines.append(f"  - {rec}")
            report_lines.append("")
            
        # Add cleaning report if available
        if hasattr(self, 'cleaning_report') and self.cleaning_report.get('actions_taken'):
            report_lines.extend([
                "CLEANING ACTIONS TAKEN:",
                "-" * 25
            ])
            for action in self.cleaning_report['actions_taken']:
                report_lines.append(f"  - {action}")
                
        return "\n".join(report_lines)