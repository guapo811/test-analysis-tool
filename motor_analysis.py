"""
Motor-Specific Analysis Module
Provides specialized analysis for BLDC motors including efficiency, torque ripple, 
harmonic analysis, and motor-specific fault detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class MotorAnalyzer:
    """Specialized motor analysis for BLDC motors."""
    
    def __init__(self, config: Dict):
        self.config = config['motor_analysis']
        self.motor_constants = {
            'pole_pairs': 4,  # Default, can be configured
            'rated_voltage': 24,  # Default rated voltage
            'rated_current': 10,  # Default rated current
            'rated_speed': 3000,  # Default rated speed (RPM)
            'rated_torque': 1.0   # Default rated torque (Nm)
        }
        
    def set_motor_parameters(self, pole_pairs: int = None, rated_voltage: float = None,
                           rated_current: float = None, rated_speed: float = None,
                           rated_torque: float = None):
        """Set motor-specific parameters for analysis."""
        if pole_pairs is not None:
            self.motor_constants['pole_pairs'] = pole_pairs
        if rated_voltage is not None:
            self.motor_constants['rated_voltage'] = rated_voltage
        if rated_current is not None:
            self.motor_constants['rated_current'] = rated_current
        if rated_speed is not None:
            self.motor_constants['rated_speed'] = rated_speed
        if rated_torque is not None:
            self.motor_constants['rated_torque'] = rated_torque
            
        logger.info(f"Motor parameters updated: {self.motor_constants}")
    
    def calculate_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate motor efficiency from power measurements.
        
        Args:
            df: DataFrame containing voltage, current, speed, and torque data
            
        Returns:
            Dictionary containing efficiency analysis results
        """
        if not self.config['enable_efficiency_calculation']:
            return {'status': 'efficiency_calculation_disabled'}
            
        results = {
            'status': 'completed',
            'efficiency_data': {},
            'efficiency_statistics': {},
            'load_efficiency_map': {}
        }
        
        try:
            # Find relevant columns
            voltage_cols = self._find_columns(df, ['voltage', 'volt', 'v_'])
            current_cols = self._find_columns(df, ['current', 'amp', 'i_'])
            speed_cols = self._find_columns(df, ['speed', 'rpm'])
            torque_cols = self._find_columns(df, ['torque'])
            power_cols = self._find_columns(df, ['power', 'watt'])
            
            if not voltage_cols or not current_cols:
                results['status'] = 'insufficient_electrical_data'
                return results
                
            # Calculate electrical power
            voltage_data = df[voltage_cols[0]].dropna()
            current_data = df[current_cols[0]].dropna()
            
            # Align data
            common_index = voltage_data.index.intersection(current_data.index)
            if len(common_index) < 10:
                results['status'] = 'insufficient_aligned_data'
                return results
                
            voltage_aligned = voltage_data.loc[common_index]
            current_aligned = current_data.loc[common_index]
            
            # Calculate electrical power (P_in = V * I for DC, assuming unity power factor)
            electrical_power = voltage_aligned * current_aligned
            
            # Calculate mechanical power if speed and torque are available
            mechanical_power = None
            if speed_cols and torque_cols:
                speed_data = df[speed_cols[0]].loc[common_index].dropna()
                torque_data = df[torque_cols[0]].loc[common_index].dropna()
                
                mech_common_index = speed_data.index.intersection(torque_data.index)
                if len(mech_common_index) > 10:
                    speed_aligned = speed_data.loc[mech_common_index]
                    torque_aligned = torque_data.loc[mech_common_index]
                    
                    # Convert RPM to rad/s and calculate mechanical power
                    angular_velocity = speed_aligned * 2 * np.pi / 60
                    mechanical_power = torque_aligned * angular_velocity
                    
                    # Align electrical and mechanical power
                    power_common_index = electrical_power.index.intersection(mechanical_power.index)
                    if len(power_common_index) > 10:
                        elec_power_aligned = electrical_power.loc[power_common_index]
                        mech_power_aligned = mechanical_power.loc[power_common_index]
                        
                        # Calculate efficiency (avoid division by zero)
                        efficiency = np.where(elec_power_aligned > 0.1, 
                                            mech_power_aligned / elec_power_aligned * 100, 
                                            0)
                        
                        results['efficiency_data'] = {
                            'electrical_power': elec_power_aligned.values,
                            'mechanical_power': mech_power_aligned.values,
                            'efficiency': efficiency,
                            'timestamps': power_common_index.tolist()
                        }
                        
                        # Calculate efficiency statistics
                        valid_efficiency = efficiency[efficiency > 0]
                        if len(valid_efficiency) > 0:
                            results['efficiency_statistics'] = {
                                'mean_efficiency': np.mean(valid_efficiency),
                                'max_efficiency': np.max(valid_efficiency),
                                'min_efficiency': np.min(valid_efficiency),
                                'std_efficiency': np.std(valid_efficiency),
                                'efficiency_range': np.max(valid_efficiency) - np.min(valid_efficiency)
                            }
                            
                        # Create load efficiency map
                        if self.config['load_mapping']['enabled']:
                            load_map = self._create_efficiency_load_map(
                                elec_power_aligned, efficiency
                            )
                            results['load_efficiency_map'] = load_map
                            
            logger.info("Motor efficiency analysis completed")
            
        except Exception as e:
            logger.error(f"Efficiency calculation failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def analyze_torque_ripple(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze torque ripple characteristics.
        
        Args:
            df: DataFrame containing torque and speed data
            
        Returns:
            Dictionary containing torque ripple analysis results
        """
        if not self.config['enable_torque_ripple_analysis']:
            return {'status': 'torque_ripple_analysis_disabled'}
            
        results = {
            'status': 'completed',
            'ripple_statistics': {},
            'harmonic_analysis': {},
            'ripple_frequency_content': {}
        }
        
        try:
            # Find torque and speed columns
            torque_cols = self._find_columns(df, ['torque'])
            speed_cols = self._find_columns(df, ['speed', 'rpm'])
            
            if not torque_cols:
                results['status'] = 'no_torque_data'
                return results
                
            torque_data = df[torque_cols[0]].dropna()
            
            if len(torque_data) < 100:
                results['status'] = 'insufficient_torque_data'
                return results
                
            # Remove DC component (mean torque)
            mean_torque = torque_data.mean()
            torque_ripple = torque_data - mean_torque
            
            # Calculate ripple statistics
            results['ripple_statistics'] = {
                'mean_torque': mean_torque,
                'rms_ripple': np.sqrt(np.mean(torque_ripple**2)),
                'peak_to_peak_ripple': torque_ripple.max() - torque_ripple.min(),
                'ripple_factor': np.std(torque_ripple) / mean_torque * 100 if mean_torque != 0 else 0,
                'max_positive_ripple': torque_ripple.max(),
                'max_negative_ripple': torque_ripple.min()
            }
            
            # Harmonic analysis of torque ripple
            if len(torque_ripple) >= 256:
                harmonic_results = self._analyze_torque_harmonics(torque_ripple, speed_cols, df)
                results['harmonic_analysis'] = harmonic_results
                
            # Frequency content analysis
            if len(torque_ripple) >= 128:
                freq_results = self._analyze_ripple_frequency_content(torque_ripple)
                results['ripple_frequency_content'] = freq_results
                
            logger.info("Torque ripple analysis completed")
            
        except Exception as e:
            logger.error(f"Torque ripple analysis failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def analyze_harmonics(self, df: pd.DataFrame, signal_type: str = 'current') -> Dict[str, Any]:
        """
        Analyze harmonic content in motor signals.
        
        Args:
            df: DataFrame containing motor signals
            signal_type: Type of signal to analyze ('current', 'voltage', 'torque')
            
        Returns:
            Dictionary containing harmonic analysis results
        """
        if not self.config['enable_harmonic_analysis']:
            return {'status': 'harmonic_analysis_disabled'}
            
        results = {
            'status': 'completed',
            'signal_type': signal_type,
            'harmonic_orders': self.config['harmonic_orders'],
            'harmonic_magnitudes': {},
            'thd_analysis': {},
            'dominant_harmonics': {}
        }
        
        try:
            # Find relevant columns based on signal type
            if signal_type == 'current':
                signal_cols = self._find_columns(df, ['current', 'amp', 'i_'])
            elif signal_type == 'voltage':
                signal_cols = self._find_columns(df, ['voltage', 'volt', 'v_'])
            elif signal_type == 'torque':
                signal_cols = self._find_columns(df, ['torque'])
            else:
                results['status'] = 'unknown_signal_type'
                return results
                
            if not signal_cols:
                results['status'] = f'no_{signal_type}_data'
                return results
                
            # Analyze harmonics for each relevant column
            for col in signal_cols[:2]:  # Limit to first 2 columns
                signal_data = df[col].dropna()
                
                if len(signal_data) < 256:
                    continue
                    
                harmonic_result = self._analyze_signal_harmonics(signal_data, col)
                results['harmonic_magnitudes'][col] = harmonic_result
                
            logger.info(f"Harmonic analysis completed for {signal_type}")
            
        except Exception as e:
            logger.error(f"Harmonic analysis failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def detect_cogging_torque(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and analyze cogging torque characteristics.
        
        Args:
            df: DataFrame containing position/angle and torque data
            
        Returns:
            Dictionary containing cogging torque analysis results
        """
        if not self.config['enable_cogging_detection']:
            return {'status': 'cogging_detection_disabled'}
            
        results = {
            'status': 'completed',
            'cogging_detected': False,
            'cogging_characteristics': {},
            'position_analysis': {}
        }
        
        try:
            # Find torque and position/angle columns
            torque_cols = self._find_columns(df, ['torque'])
            position_cols = self._find_columns(df, ['position', 'angle', 'theta'])
            
            if not torque_cols:
                results['status'] = 'no_torque_data'
                return results
                
            torque_data = df[torque_cols[0]].dropna()
            
            # If no position data, try to infer from time-based patterns
            if position_cols:
                position_data = df[position_cols[0]].dropna()
                common_index = torque_data.index.intersection(position_data.index)
                
                if len(common_index) > 100:
                    torque_aligned = torque_data.loc[common_index]
                    position_aligned = position_data.loc[common_index]
                    
                    cogging_analysis = self._analyze_cogging_with_position(
                        torque_aligned, position_aligned
                    )
                    results.update(cogging_analysis)
                else:
                    # Fall back to time-based analysis
                    cogging_analysis = self._analyze_cogging_time_based(torque_data)
                    results.update(cogging_analysis)
            else:
                # Time-based cogging analysis
                cogging_analysis = self._analyze_cogging_time_based(torque_data)
                results.update(cogging_analysis)
                
            logger.info("Cogging torque analysis completed")
            
        except Exception as e:
            logger.error(f"Cogging torque analysis failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def analyze_temperature_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlation between temperature and motor performance.
        
        Args:
            df: DataFrame containing temperature and performance data
            
        Returns:
            Dictionary containing temperature correlation analysis
        """
        if not self.config['temperature_correlation']:
            return {'status': 'temperature_correlation_disabled'}
            
        results = {
            'status': 'completed',
            'temperature_effects': {},
            'correlations': {},
            'thermal_analysis': {}
        }
        
        try:
            # Find temperature columns
            temp_cols = self._find_columns(df, ['temp', 'temperature', 'thermal'])
            
            if not temp_cols:
                results['status'] = 'no_temperature_data'
                return results
                
            # Find performance columns
            perf_cols = self._find_columns(df, ['efficiency', 'power', 'current', 'speed', 'torque'])
            
            if not perf_cols:
                results['status'] = 'no_performance_data'
                return results
                
            for temp_col in temp_cols[:2]:  # Analyze first 2 temperature sensors
                temp_data = df[temp_col].dropna()
                
                temp_correlations = {}
                for perf_col in perf_cols:
                    perf_data = df[perf_col].dropna()
                    
                    # Find common data points
                    common_index = temp_data.index.intersection(perf_data.index)
                    
                    if len(common_index) > 20:
                        temp_aligned = temp_data.loc[common_index]
                        perf_aligned = perf_data.loc[common_index]
                        
                        # Calculate correlation
                        correlation = temp_aligned.corr(perf_aligned)
                        
                        if not np.isnan(correlation):
                            temp_correlations[perf_col] = {
                                'correlation': correlation,
                                'strength': abs(correlation),
                                'direction': 'positive' if correlation > 0 else 'negative'
                            }
                            
                results['correlations'][temp_col] = temp_correlations
                
                # Thermal analysis
                thermal_stats = {
                    'mean_temperature': temp_data.mean(),
                    'max_temperature': temp_data.max(),
                    'min_temperature': temp_data.min(),
                    'temperature_range': temp_data.max() - temp_data.min(),
                    'temperature_std': temp_data.std()
                }
                
                results['thermal_analysis'][temp_col] = thermal_stats
                
            logger.info("Temperature correlation analysis completed")
            
        except Exception as e:
            logger.error(f"Temperature correlation analysis failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def _find_columns(self, df: pd.DataFrame, keywords: List[str]) -> List[str]:
        """Find columns containing specific keywords."""
        found_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                found_columns.append(col)
        return found_columns
    
    def _create_efficiency_load_map(self, power_data: pd.Series, efficiency_data: np.ndarray) -> Dict:
        """Create efficiency vs load mapping."""
        load_ranges = self.config['load_mapping']['load_ranges']
        efficiency_targets = self.config['load_mapping']['efficiency_targets']
        
        # Calculate load percentage based on maximum power
        max_power = power_data.max()
        load_percentage = (power_data / max_power) * 100
        
        load_map = {}
        for i, load_range in enumerate(load_ranges[:-1]):
            next_range = load_ranges[i + 1]
            
            # Find data points in this load range
            mask = (load_percentage >= load_range) & (load_percentage < next_range)
            range_efficiency = efficiency_data[mask]
            
            if len(range_efficiency) > 0:
                load_map[f'{load_range}-{next_range}%'] = {
                    'mean_efficiency': np.mean(range_efficiency),
                    'max_efficiency': np.max(range_efficiency),
                    'min_efficiency': np.min(range_efficiency),
                    'data_points': len(range_efficiency),
                    'target_efficiency': efficiency_targets[i] if i < len(efficiency_targets) else None
                }
                
        return load_map
    
    def _analyze_torque_harmonics(self, torque_ripple: pd.Series, speed_cols: List[str], df: pd.DataFrame) -> Dict:
        """Analyze harmonic content of torque ripple."""
        harmonic_orders = self.config['torque_ripple_harmonics']
        
        # Perform FFT analysis
        N = len(torque_ripple)
        fft_values = fft(torque_ripple.values)
        
        # If speed data is available, perform order analysis
        if speed_cols:
            speed_data = df[speed_cols[0]].loc[torque_ripple.index].dropna()
            
            if len(speed_data) > N // 2:
                # Convert to order domain
                mean_speed = speed_data.mean()
                fundamental_freq = mean_speed / 60.0  # Hz
                
                # Calculate order amplitudes
                order_amplitudes = {}
                frequencies = fftfreq(N, 1.0)[:N//2]
                magnitude = np.abs(fft_values[:N//2]) * 2/N
                
                for order in harmonic_orders:
                    target_freq = fundamental_freq * order
                    if target_freq < frequencies[-1]:  # Within Nyquist limit
                        closest_idx = np.argmin(np.abs(frequencies - target_freq))
                        order_amplitudes[f'order_{order}'] = magnitude[closest_idx]
                        
                return {
                    'order_amplitudes': order_amplitudes,
                    'fundamental_frequency': fundamental_freq,
                    'total_harmonic_distortion': np.sqrt(sum([amp**2 for amp in order_amplitudes.values()])) / order_amplitudes.get('order_1', 1)
                }
                
        # Fall back to regular FFT analysis
        frequencies = fftfreq(N, 1.0)[:N//2]
        magnitude = np.abs(fft_values[:N//2]) * 2/N
        
        # Find dominant frequencies
        peak_indices = np.argsort(magnitude)[-5:][::-1]  # Top 5 peaks
        dominant_freqs = frequencies[peak_indices]
        dominant_mags = magnitude[peak_indices]
        
        return {
            'dominant_frequencies': dominant_freqs.tolist(),
            'dominant_magnitudes': dominant_mags.tolist(),
            'frequency_spectrum': {
                'frequencies': frequencies.tolist(),
                'magnitudes': magnitude.tolist()
            }
        }
    
    def _analyze_ripple_frequency_content(self, torque_ripple: pd.Series) -> Dict:
        """Analyze frequency content of torque ripple."""
        # Perform FFT
        N = len(torque_ripple)
        fft_values = fft(torque_ripple.values)
        frequencies = fftfreq(N, 1.0)[:N//2]
        magnitude = np.abs(fft_values[:N//2]) * 2/N
        
        # Calculate power spectral density
        psd = magnitude**2
        
        # Find peaks in the spectrum
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        
        return {
            'peak_frequencies': frequencies[peaks].tolist(),
            'peak_magnitudes': magnitude[peaks].tolist(),
            'total_power': np.sum(psd),
            'frequency_centroid': np.sum(frequencies * magnitude) / np.sum(magnitude)
        }
    
    def _analyze_signal_harmonics(self, signal_data: pd.Series, column_name: str) -> Dict:
        """Analyze harmonics in a signal."""
        harmonic_orders = self.config['harmonic_orders']
        
        # Remove DC component
        signal_ac = signal_data - signal_data.mean()
        
        # Perform FFT
        N = len(signal_ac)
        fft_values = fft(signal_ac.values)
        frequencies = fftfreq(N, 1.0)[:N//2]
        magnitude = np.abs(fft_values[:N//2]) * 2/N
        
        # Find fundamental frequency (largest peak)
        fundamental_idx = np.argmax(magnitude)
        fundamental_freq = frequencies[fundamental_idx]
        fundamental_mag = magnitude[fundamental_idx]
        
        # Calculate harmonic magnitudes
        harmonic_mags = {}
        for order in harmonic_orders:
            harmonic_freq = fundamental_freq * order
            
            if harmonic_freq < frequencies[-1]:  # Within Nyquist limit
                # Find closest frequency bin
                closest_idx = np.argmin(np.abs(frequencies - harmonic_freq))
                harmonic_mags[f'harmonic_{order}'] = magnitude[closest_idx]
                
        # Calculate Total Harmonic Distortion (THD)
        harmonic_sum = sum([mag**2 for order, mag in harmonic_mags.items() if order != 'harmonic_1'])
        thd = np.sqrt(harmonic_sum) / fundamental_mag * 100 if fundamental_mag > 0 else 0
        
        return {
            'fundamental_frequency': fundamental_freq,
            'fundamental_magnitude': fundamental_mag,
            'harmonic_magnitudes': harmonic_mags,
            'thd_percent': thd,
            'frequency_spectrum': {
                'frequencies': frequencies.tolist(),
                'magnitudes': magnitude.tolist()
            }
        }
    
    def _analyze_cogging_with_position(self, torque_data: pd.Series, position_data: pd.Series) -> Dict:
        """Analyze cogging torque using position data."""
        # Sort by position
        combined_data = pd.DataFrame({'torque': torque_data, 'position': position_data})
        combined_data = combined_data.sort_values('position')
        
        # Look for periodic patterns in torque vs position
        position_range = position_data.max() - position_data.min()
        
        # Expected cogging frequency based on pole pairs and slots
        pole_pairs = self.motor_constants['pole_pairs']
        expected_cogging_periods = [
            360 / (pole_pairs * 6),  # Common slot/pole combinations
            360 / (pole_pairs * 9),
            360 / (pole_pairs * 12)
        ]
        
        cogging_detected = False
        cogging_magnitude = 0
        
        # Analyze torque variation vs position
        if position_range > 180:  # At least half revolution
            # Remove trend
            torque_detrended = combined_data['torque'] - combined_data['torque'].rolling(window=10).mean()
            torque_detrended = torque_detrended.dropna()
            
            cogging_magnitude = torque_detrended.std()
            
            # Check if variation is significant
            if cogging_magnitude > torque_data.mean() * 0.01:  # 1% of mean torque
                cogging_detected = True
                
        return {
            'cogging_detected': cogging_detected,
            'cogging_characteristics': {
                'magnitude': cogging_magnitude,
                'position_dependent': True,
                'analysis_range_degrees': position_range
            }
        }
    
    def _analyze_cogging_time_based(self, torque_data: pd.Series) -> Dict:
        """Analyze cogging torque using time-based data."""
        # Look for periodic patterns in torque
        torque_detrended = torque_data - torque_data.rolling(window=20).mean()
        torque_detrended = torque_detrended.dropna()
        
        if len(torque_detrended) < 100:
            return {
                'cogging_detected': False,
                'cogging_characteristics': {'insufficient_data': True}
            }
            
        # Perform FFT to find periodic components
        N = len(torque_detrended)
        fft_values = fft(torque_detrended.values)
        frequencies = fftfreq(N, 1.0)[:N//2]
        magnitude = np.abs(fft_values[:N//2]) * 2/N
        
        # Find dominant low-frequency components (cogging is typically low frequency)
        low_freq_mask = frequencies < 0.1  # Adjust based on expected cogging frequency
        low_freq_magnitude = magnitude[low_freq_mask]
        
        cogging_detected = False
        cogging_magnitude = torque_detrended.std()
        
        if len(low_freq_magnitude) > 0 and np.max(low_freq_magnitude) > cogging_magnitude * 0.1:
            cogging_detected = True
            
        return {
            'cogging_detected': cogging_detected,
            'cogging_characteristics': {
                'magnitude': cogging_magnitude,
                'time_based_analysis': True,
                'dominant_frequency': frequencies[np.argmax(magnitude)] if len(magnitude) > 0 else 0
            }
        }