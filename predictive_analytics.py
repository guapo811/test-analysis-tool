"""
Predictive Analytics Module for Motor Data Analysis
Provides anomaly detection, trend analysis, and pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy import stats
from scipy.signal import find_peaks
import joblib
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

class PredictiveAnalyzer:
    """Advanced predictive analytics for motor data analysis."""
    
    def __init__(self, config: Dict):
        self.config = config['predictive_analytics']
        self.models = {}
        self.scalers = {}
        self.baseline_stats = {}
        
    def detect_anomalies(self, df: pd.DataFrame, method: str = None) -> Dict[str, Any]:
        """
        Detect anomalies in the data using various methods.
        
        Args:
            df: Input dataframe
            method: Anomaly detection method ('isolation_forest', 'one_class_svm', 'statistical')
        
        Returns:
            Dictionary containing anomaly detection results
        """
        if not self.config['enable_anomaly_detection']:
            return {'status': 'anomaly_detection_disabled'}
            
        if method is None:
            methods = self.config['anomaly_methods']
        else:
            methods = [method]
            
        results = {
            'status': 'completed',
            'methods_used': methods,
            'anomaly_results': {},
            'summary': {}
        }
        
        # Prepare data for anomaly detection
        numeric_data = df.select_dtypes(include=[np.number]).dropna()
        if len(numeric_data) < 10:
            results['status'] = 'insufficient_data'
            return results
            
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            contamination = self.config['contamination_rate']
            all_anomalies = np.zeros(len(numeric_data), dtype=bool)
            
            for method_name in methods:
                method_results = self._apply_anomaly_method(
                    scaled_data, method_name, contamination
                )
                
                if method_results is not None:
                    results['anomaly_results'][method_name] = {
                        'anomaly_indices': method_results['anomalies'],
                        'anomaly_count': len(method_results['anomalies']),
                        'anomaly_percentage': (len(method_results['anomalies']) / len(numeric_data)) * 100,
                        'scores': method_results.get('scores', [])
                    }
                    
                    # Combine anomalies from all methods
                    method_anomalies = np.zeros(len(numeric_data), dtype=bool)
                    method_anomalies[method_results['anomalies']] = True
                    all_anomalies |= method_anomalies
                    
            # Generate summary
            total_anomalies = np.sum(all_anomalies)
            results['summary'] = {
                'total_anomalies': int(total_anomalies),
                'total_percentage': (total_anomalies / len(numeric_data)) * 100,
                'anomaly_indices': np.where(all_anomalies)[0].tolist(),
                'normal_data_points': int(len(numeric_data) - total_anomalies)
            }
            
            # Analyze anomaly patterns
            if total_anomalies > 0:
                anomaly_analysis = self._analyze_anomaly_patterns(
                    numeric_data, all_anomalies
                )
                results['anomaly_patterns'] = anomaly_analysis
                
            logger.info(f"Anomaly detection completed. Found {total_anomalies} anomalies ({results['summary']['total_percentage']:.2f}%)")
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def _apply_anomaly_method(self, data: np.ndarray, method: str, contamination: float) -> Optional[Dict]:
        """Apply specific anomaly detection method."""
        try:
            if method == 'isolation_forest':
                model = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
                predictions = model.fit_predict(data)
                scores = model.score_samples(data)
                anomalies = np.where(predictions == -1)[0]
                
                return {
                    'anomalies': anomalies,
                    'scores': scores,
                    'model': model
                }
                
            elif method == 'one_class_svm':
                model = OneClassSVM(
                    nu=contamination,
                    kernel='rbf',
                    gamma='scale'
                )
                predictions = model.fit_predict(data)
                scores = model.score_samples(data)
                anomalies = np.where(predictions == -1)[0]
                
                return {
                    'anomalies': anomalies,
                    'scores': scores,
                    'model': model
                }
                
            elif method == 'statistical':
                # Use statistical methods (Z-score and IQR)
                anomalies = set()
                
                for i, col_data in enumerate(data.T):
                    # Z-score method
                    z_scores = np.abs(stats.zscore(col_data))
                    z_anomalies = np.where(z_scores > 3)[0]
                    anomalies.update(z_anomalies)
                    
                    # IQR method
                    Q1 = np.percentile(col_data, 25)
                    Q3 = np.percentile(col_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    iqr_anomalies = np.where((col_data < lower_bound) | (col_data > upper_bound))[0]
                    anomalies.update(iqr_anomalies)
                    
                return {
                    'anomalies': list(anomalies),
                    'scores': None
                }
                
        except Exception as e:
            logger.error(f"Failed to apply {method} anomaly detection: {e}")
            return None
    
    def _analyze_anomaly_patterns(self, data: pd.DataFrame, anomaly_mask: np.ndarray) -> Dict:
        """Analyze patterns in detected anomalies."""
        anomaly_data = data[anomaly_mask]
        normal_data = data[~anomaly_mask]
        
        patterns = {
            'temporal_clustering': {},
            'feature_analysis': {},
            'severity_analysis': {}
        }
        
        # Temporal clustering analysis
        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) > 1:
            # Check for consecutive anomalies
            consecutive_groups = []
            current_group = [anomaly_indices[0]]
            
            for i in range(1, len(anomaly_indices)):
                if anomaly_indices[i] - anomaly_indices[i-1] == 1:
                    current_group.append(anomaly_indices[i])
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [anomaly_indices[i]]
                    
            if len(current_group) > 1:
                consecutive_groups.append(current_group)
                
            patterns['temporal_clustering'] = {
                'consecutive_groups': len(consecutive_groups),
                'largest_group_size': max([len(group) for group in consecutive_groups]) if consecutive_groups else 1,
                'isolated_anomalies': len(anomaly_indices) - sum([len(group) for group in consecutive_groups])
            }
            
        # Feature analysis
        for col in data.columns:
            if len(anomaly_data) > 0:
                anomaly_mean = anomaly_data[col].mean()
                normal_mean = normal_data[col].mean()
                
                patterns['feature_analysis'][col] = {
                    'anomaly_mean': anomaly_mean,
                    'normal_mean': normal_mean,
                    'deviation_factor': abs(anomaly_mean - normal_mean) / normal_data[col].std() if normal_data[col].std() > 0 else 0
                }
                
        return patterns
    
    def trend_analysis(self, df: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, Any]:
        """
        Perform trend analysis on time series data.
        
        Args:
            df: Input dataframe
            target_columns: Columns to analyze (if None, analyze all numeric columns)
        
        Returns:
            Dictionary containing trend analysis results
        """
        if not self.config['enable_trend_analysis']:
            return {'status': 'trend_analysis_disabled'}
            
        if target_columns is None:
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        window_size = self.config['trend_window_size']
        
        results = {
            'status': 'completed',
            'trend_results': {},
            'summary': {}
        }
        
        try:
            trends_detected = 0
            
            for col in target_columns:
                data = df[col].dropna()
                if len(data) < window_size:
                    continue
                    
                trend_result = self._analyze_column_trend(data, window_size)
                results['trend_results'][col] = trend_result
                
                if trend_result['trend_detected']:
                    trends_detected += 1
                    
            results['summary'] = {
                'columns_analyzed': len(results['trend_results']),
                'trends_detected': trends_detected,
                'trend_percentage': (trends_detected / len(results['trend_results'])) * 100 if results['trend_results'] else 0
            }
            
            logger.info(f"Trend analysis completed. Trends detected in {trends_detected} columns")
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def _analyze_column_trend(self, data: pd.Series, window_size: int) -> Dict:
        """Analyze trend for a single column."""
        trend_result = {
            'trend_detected': False,
            'trend_direction': 'none',
            'trend_strength': 0.0,
            'change_points': [],
            'linear_trend': {},
            'seasonal_components': {}
        }
        
        try:
            # Linear trend analysis
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            
            trend_result['linear_trend'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Determine trend direction and strength
            if abs(r_value) > 0.3 and p_value < 0.05:
                trend_result['trend_detected'] = True
                trend_result['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
                trend_result['trend_strength'] = abs(r_value)
                
            # Change point detection using moving averages
            if len(data) >= window_size * 2:
                change_points = self._detect_change_points(data, window_size)
                trend_result['change_points'] = change_points
                
            # Simple seasonal analysis
            if len(data) >= window_size * 4:
                seasonal_info = self._detect_seasonality(data, window_size)
                trend_result['seasonal_components'] = seasonal_info
                
        except Exception as e:
            logger.error(f"Column trend analysis failed: {e}")
            
        return trend_result
    
    def _detect_change_points(self, data: pd.Series, window_size: int) -> List[int]:
        """Detect change points in the data."""
        change_points = []
        
        try:
            # Use moving average to detect significant changes
            moving_avg = data.rolling(window=window_size).mean().dropna()
            moving_std = data.rolling(window=window_size).std().dropna()
            
            # Look for points where the mean changes significantly
            for i in range(window_size, len(moving_avg) - window_size):
                before_mean = moving_avg.iloc[i-window_size//2:i+1].mean()
                after_mean = moving_avg.iloc[i:i+window_size//2].mean()
                
                # Use the pooled standard deviation
                pooled_std = (moving_std.iloc[i-window_size//2:i+window_size//2]).mean()
                
                if pooled_std > 0:
                    # Significant change if difference is more than 2 standard deviations
                    if abs(after_mean - before_mean) > 2 * pooled_std:
                        change_points.append(i)
                        
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            
        return change_points
    
    def _detect_seasonality(self, data: pd.Series, period: int) -> Dict:
        """Detect seasonal patterns in the data."""
        seasonal_info = {
            'seasonal_detected': False,
            'period': period,
            'seasonal_strength': 0.0
        }
        
        try:
            # Simple seasonal decomposition using autocorrelation
            if len(data) >= period * 3:
                # Calculate autocorrelation at the suspected period
                autocorr = data.autocorr(lag=period)
                
                if abs(autocorr) > 0.3:  # Threshold for seasonal detection
                    seasonal_info['seasonal_detected'] = True
                    seasonal_info['seasonal_strength'] = abs(autocorr)
                    
        except Exception as e:
            logger.error(f"Seasonality detection failed: {e}")
            
        return seasonal_info
    
    def pattern_recognition(self, df: pd.DataFrame, pattern_type: str = 'vibration') -> Dict[str, Any]:
        """
        Recognize patterns in motor data.
        
        Args:
            df: Input dataframe
            pattern_type: Type of pattern to recognize ('vibration', 'performance', 'fault')
        
        Returns:
            Dictionary containing pattern recognition results
        """
        if not self.config['enable_pattern_recognition']:
            return {'status': 'pattern_recognition_disabled'}
            
        results = {
            'status': 'completed',
            'pattern_type': pattern_type,
            'patterns_found': {},
            'confidence_scores': {}
        }
        
        try:
            if pattern_type == 'vibration':
                results.update(self._recognize_vibration_patterns(df))
            elif pattern_type == 'performance':
                results.update(self._recognize_performance_patterns(df))
            elif pattern_type == 'fault':
                results.update(self._recognize_fault_patterns(df))
                
            logger.info(f"Pattern recognition completed for {pattern_type}")
            
        except Exception as e:
            logger.error(f"Pattern recognition failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def _recognize_vibration_patterns(self, df: pd.DataFrame) -> Dict:
        """Recognize vibration-specific patterns."""
        patterns = {}
        
        # Find vibration columns
        vib_columns = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['vib', 'accel', 'shake']
        )]
        
        for col in vib_columns:
            data = df[col].dropna()
            if len(data) < 100:
                continue
                
            col_patterns = {}
            
            # Detect periodic patterns
            try:
                # Find peaks in the signal
                peaks, properties = find_peaks(data, height=data.std(), distance=10)
                
                if len(peaks) > 5:
                    # Analyze peak intervals
                    peak_intervals = np.diff(peaks)
                    avg_interval = np.mean(peak_intervals)
                    interval_std = np.std(peak_intervals)
                    
                    # Regular periodic pattern if intervals are consistent
                    if interval_std / avg_interval < 0.3:
                        col_patterns['periodic_pattern'] = {
                            'detected': True,
                            'average_period': avg_interval,
                            'regularity': 1 - (interval_std / avg_interval),
                            'peak_count': len(peaks)
                        }
                    else:
                        col_patterns['periodic_pattern'] = {'detected': False}
                        
                # Detect amplitude modulation
                envelope_data = np.abs(data)
                envelope_peaks, _ = find_peaks(envelope_data, distance=50)
                
                if len(envelope_peaks) > 3:
                    col_patterns['amplitude_modulation'] = {
                        'detected': True,
                        'modulation_frequency': len(envelope_peaks) / len(data),
                        'peak_count': len(envelope_peaks)
                    }
                else:
                    col_patterns['amplitude_modulation'] = {'detected': False}
                    
            except Exception as e:
                logger.error(f"Vibration pattern analysis failed for {col}: {e}")
                
            patterns[col] = col_patterns
            
        return {'vibration_patterns': patterns}
    
    def _recognize_performance_patterns(self, df: pd.DataFrame) -> Dict:
        """Recognize performance-related patterns."""
        patterns = {}
        
        # Find performance-related columns
        perf_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in 
                  ['speed', 'rpm', 'torque', 'power', 'efficiency', 'current', 'voltage']):
                perf_columns.append(col)
                
        # Analyze relationships between performance parameters
        if len(perf_columns) >= 2:
            correlation_matrix = df[perf_columns].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'var1': correlation_matrix.columns[i],
                            'var2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'relationship': 'positive' if corr_value > 0 else 'negative'
                        })
                        
            patterns['performance_correlations'] = strong_correlations
            
        return {'performance_patterns': patterns}
    
    def _recognize_fault_patterns(self, df: pd.DataFrame) -> Dict:
        """Recognize fault-indicative patterns."""
        patterns = {}
        
        # Look for sudden changes in key parameters
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            data = df[col].dropna()
            if len(data) < 50:
                continue
                
            # Detect sudden jumps
            diff_data = data.diff().abs()
            threshold = diff_data.quantile(0.95)  # 95th percentile
            
            sudden_changes = diff_data[diff_data > threshold]
            
            if len(sudden_changes) > 0:
                patterns[col] = {
                    'sudden_changes': {
                        'count': len(sudden_changes),
                        'max_change': sudden_changes.max(),
                        'change_indices': sudden_changes.index.tolist()
                    }
                }
                
        return {'fault_patterns': patterns}
    
    def train_predictive_model(self, df: pd.DataFrame, target_column: str, 
                             model_type: str = 'anomaly') -> Dict[str, Any]:
        """
        Train a predictive model for future anomaly detection or forecasting.
        
        Args:
            df: Training dataframe
            target_column: Column to predict
            model_type: Type of model ('anomaly', 'regression')
        
        Returns:
            Dictionary containing model training results
        """
        if not self.config['model_training']['auto_train']:
            return {'status': 'auto_training_disabled'}
            
        results = {
            'status': 'completed',
            'model_type': model_type,
            'target_column': target_column,
            'training_metrics': {}
        }
        
        try:
            # Prepare data
            features = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
            target = df[target_column].dropna()
            
            # Align features and target
            common_index = features.index.intersection(target.index)
            X = features.loc[common_index]
            y = target.loc[common_index]
            
            if len(X) < 50:
                results['status'] = 'insufficient_data'
                return results
                
            # Split data
            validation_split = self.config['model_training']['validation_split']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if model_type == 'anomaly':
                # Train anomaly detection model
                model = IsolationForest(random_state=42, contamination=0.1)
                model.fit(X_train_scaled)
                
                # Evaluate
                train_predictions = model.predict(X_train_scaled)
                test_predictions = model.predict(X_test_scaled)
                
                results['training_metrics'] = {
                    'train_anomaly_rate': (train_predictions == -1).mean(),
                    'test_anomaly_rate': (test_predictions == -1).mean()
                }
                
            # Store model and scaler
            model_key = f"{target_column}_{model_type}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Save model to disk if configured
            try:
                joblib.dump(model, f"model_{model_key}.pkl")
                joblib.dump(scaler, f"scaler_{model_key}.pkl")
                results['model_saved'] = True
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
                results['model_saved'] = False
                
            logger.info(f"Model training completed for {target_column}")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def predict_future_anomalies(self, df: pd.DataFrame, model_key: str) -> Dict[str, Any]:
        """
        Use trained model to predict future anomalies.
        
        Args:
            df: New data for prediction
            model_key: Key of the trained model to use
        
        Returns:
            Dictionary containing prediction results
        """
        if model_key not in self.models:
            return {'status': 'model_not_found', 'available_models': list(self.models.keys())}
            
        results = {
            'status': 'completed',
            'model_used': model_key,
            'predictions': {}
        }
        
        try:
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Prepare data
            features = df.select_dtypes(include=[np.number])
            scaled_features = scaler.transform(features)
            
            # Make predictions
            predictions = model.predict(scaled_features)
            scores = model.score_samples(scaled_features)
            
            anomaly_indices = np.where(predictions == -1)[0]
            
            results['predictions'] = {
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': (len(anomaly_indices) / len(predictions)) * 100,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': scores[anomaly_indices].tolist(),
                'all_scores': scores.tolist()
            }
            
            logger.info(f"Prediction completed. Found {len(anomaly_indices)} potential anomalies")
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results