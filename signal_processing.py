"""
Advanced Signal Processing Module for Motor Data Analysis
Provides FFT analysis, filtering, envelope analysis, and spectral analysis.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class SignalProcessor:
    """Advanced signal processing for motor vibration and performance data."""
    
    def __init__(self, config: Dict):
        self.config = config['signal_processing']
        self.sampling_rate = None
        
    def set_sampling_rate(self, fs: float):
        """Set sampling rate for signal processing operations."""
        self.sampling_rate = fs
        
    def estimate_sampling_rate(self, time_data: pd.Series) -> float:
        """Estimate sampling rate from time data."""
        if len(time_data) < 2:
            return 1000.0  # Default fallback
            
        time_diff = time_data.diff().dropna()
        # Use median to avoid outliers
        dt = time_diff.median()
        
        if dt > 0:
            fs = 1.0 / dt
        else:
            fs = 1000.0  # Default fallback
            
        logger.info(f"Estimated sampling rate: {fs:.2f} Hz")
        return fs
    
    def apply_window(self, data: np.ndarray, window_type: str = None) -> np.ndarray:
        """Apply windowing function to data."""
        if window_type is None:
            window_type = self.config['fft_window']
            
        if window_type == 'hann':
            window = signal.hann(len(data))
        elif window_type == 'hamming':
            window = signal.hamming(len(data))
        elif window_type == 'blackman':
            window = signal.blackman(len(data))
        elif window_type == 'bartlett':
            window = signal.bartlett(len(data))
        else:
            window = np.ones(len(data))  # Rectangular window
            
        return data * window
    
    def compute_fft(self, data: pd.Series, time_data: pd.Series = None) -> Dict:
        """
        Compute FFT analysis of the signal.
        
        Returns:
            Dictionary containing frequencies, magnitudes, phases, and PSD
        """
        if not self.config['enable_fft_analysis']:
            return {}
            
        # Clean data
        clean_data = data.dropna().values
        if len(clean_data) < 4:
            logger.warning("Insufficient data for FFT analysis")
            return {}
            
        # Estimate sampling rate if not provided
        if self.sampling_rate is None and time_data is not None:
            self.sampling_rate = self.estimate_sampling_rate(time_data)
        elif self.sampling_rate is None:
            self.sampling_rate = 1000.0  # Default
            
        # Apply windowing
        windowed_data = self.apply_window(clean_data)
        
        # Compute FFT
        N = len(windowed_data)
        fft_values = fft(windowed_data)
        frequencies = fftfreq(N, 1/self.sampling_rate)[:N//2]
        
        # Magnitude and phase
        magnitude = np.abs(fft_values[:N//2]) * 2/N
        phase = np.angle(fft_values[:N//2])
        
        # Power Spectral Density
        psd = magnitude**2
        
        # Find dominant frequencies
        dominant_indices = np.argsort(magnitude)[-5:][::-1]  # Top 5 frequencies
        dominant_freqs = frequencies[dominant_indices]
        dominant_mags = magnitude[dominant_indices]
        
        return {
            'frequencies': frequencies,
            'magnitude': magnitude,
            'phase': phase,
            'psd': psd,
            'dominant_frequencies': dominant_freqs,
            'dominant_magnitudes': dominant_mags,
            'total_power': np.sum(psd),
            'peak_frequency': frequencies[np.argmax(magnitude)],
            'peak_magnitude': np.max(magnitude)
        }
    
    def apply_filter(self, data: pd.Series, filter_type: str, cutoff_freq: float, 
                    order: int = None) -> pd.Series:
        """
        Apply digital filter to the signal.
        
        Args:
            data: Input signal
            filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
            cutoff_freq: Cutoff frequency (or [low, high] for bandpass/bandstop)
            order: Filter order
        """
        if not self.config['enable_filtering']:
            return data
            
        if order is None:
            order = self.config['default_filter_order']
            
        clean_data = data.dropna().values
        if len(clean_data) < order * 3:
            logger.warning("Insufficient data for filtering")
            return data
            
        try:
            # Normalize frequency (Nyquist frequency = 1)
            nyquist = self.sampling_rate / 2
            
            if filter_type in ['lowpass', 'highpass']:
                normalized_cutoff = cutoff_freq / nyquist
                b, a = signal.butter(order, normalized_cutoff, btype=filter_type)
            elif filter_type in ['bandpass', 'bandstop']:
                if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                    low_freq, high_freq = cutoff_freq
                    normalized_cutoff = [low_freq / nyquist, high_freq / nyquist]
                    b, a = signal.butter(order, normalized_cutoff, btype=filter_type)
                else:
                    raise ValueError("Bandpass/bandstop filters require [low, high] cutoff frequencies")
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
                
            # Apply filter
            filtered_data = signal.filtfilt(b, a, clean_data)
            
            # Create result series with original index
            result = data.copy()
            result.iloc[:len(filtered_data)] = filtered_data
            
            return result
            
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return data
    
    def envelope_analysis(self, data: pd.Series) -> Dict:
        """
        Perform envelope analysis using Hilbert transform.
        
        Returns:
            Dictionary containing envelope signal and envelope spectrum
        """
        if not self.config['enable_envelope_analysis']:
            return {}
            
        clean_data = data.dropna().values
        if len(clean_data) < 4:
            return {}
            
        try:
            # Compute analytical signal using Hilbert transform
            analytic_signal = signal.hilbert(clean_data)
            
            # Extract envelope (magnitude of analytic signal)
            envelope = np.abs(analytic_signal)
            
            # Compute envelope spectrum
            envelope_fft = fft(envelope)
            N = len(envelope)
            envelope_freqs = fftfreq(N, 1/self.sampling_rate)[:N//2]
            envelope_magnitude = np.abs(envelope_fft[:N//2]) * 2/N
            
            # Find envelope peaks
            peaks, _ = signal.find_peaks(envelope_magnitude, height=np.max(envelope_magnitude) * 0.1)
            peak_freqs = envelope_freqs[peaks]
            peak_mags = envelope_magnitude[peaks]
            
            return {
                'envelope': envelope,
                'envelope_frequencies': envelope_freqs,
                'envelope_magnitude': envelope_magnitude,
                'envelope_peaks_freq': peak_freqs,
                'envelope_peaks_mag': peak_mags,
                'envelope_rms': np.sqrt(np.mean(envelope**2))
            }
            
        except Exception as e:
            logger.error(f"Envelope analysis failed: {e}")
            return {}
    
    def spectrogram_analysis(self, data: pd.Series, nperseg: int = None, 
                           overlap: float = None) -> Dict:
        """
        Compute spectrogram for time-frequency analysis.
        
        Returns:
            Dictionary containing time, frequency, and spectrogram data
        """
        if nperseg is None:
            nperseg = self.config['fft_nperseg']
        if overlap is None:
            overlap = self.config['fft_overlap']
            
        clean_data = data.dropna().values
        if len(clean_data) < nperseg:
            return {}
            
        try:
            noverlap = int(nperseg * overlap)
            frequencies, times, Sxx = signal.spectrogram(
                clean_data, 
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            return {
                'time_bins': times,
                'frequency_bins': frequencies,
                'spectrogram': Sxx,
                'peak_frequency_vs_time': frequencies[np.argmax(Sxx, axis=0)],
                'total_power_vs_time': np.sum(Sxx, axis=0)
            }
            
        except Exception as e:
            logger.error(f"Spectrogram analysis failed: {e}")
            return {}
    
    def order_analysis(self, data: pd.Series, rpm_data: pd.Series, 
                      orders: List[float] = None) -> Dict:
        """
        Perform order analysis (RPM-based frequency analysis).
        
        Args:
            data: Vibration signal
            rpm_data: RPM signal
            orders: List of orders to analyze (e.g., [1, 2, 3] for 1X, 2X, 3X)
        """
        if orders is None:
            orders = [1, 2, 3, 4, 5, 6]
            
        try:
            # Resample to common index
            common_index = data.dropna().index.intersection(rpm_data.dropna().index)
            if len(common_index) < 10:
                return {}
                
            vibration = data.loc[common_index]
            rpm = rpm_data.loc[common_index]
            
            # Convert RPM to fundamental frequency (Hz)
            fundamental_freq = rpm / 60.0  # Hz
            
            # Calculate order frequencies
            order_results = {}
            for order in orders:
                order_freq = fundamental_freq * order
                
                # Extract amplitude at order frequency using FFT
                fft_result = self.compute_fft(vibration)
                if fft_result:
                    frequencies = fft_result['frequencies']
                    magnitude = fft_result['magnitude']
                    
                    # Find closest frequency bins to order frequencies
                    order_amplitudes = []
                    for target_freq in order_freq:
                        if target_freq <= frequencies[-1]:  # Within Nyquist limit
                            closest_idx = np.argmin(np.abs(frequencies - target_freq))
                            order_amplitudes.append(magnitude[closest_idx])
                        else:
                            order_amplitudes.append(0.0)
                    
                    order_results[f'order_{order}X'] = {
                        'frequencies': order_freq.values,
                        'amplitudes': np.array(order_amplitudes),
                        'rms_amplitude': np.sqrt(np.mean(np.array(order_amplitudes)**2))
                    }
            
            return order_results
            
        except Exception as e:
            logger.error(f"Order analysis failed: {e}")
            return {}
    
    def coherence_analysis(self, signal1: pd.Series, signal2: pd.Series) -> Dict:
        """
        Compute coherence between two signals.
        """
        try:
            # Align signals
            common_index = signal1.dropna().index.intersection(signal2.dropna().index)
            if len(common_index) < 10:
                return {}
                
            s1 = signal1.loc[common_index].values
            s2 = signal2.loc[common_index].values
            
            # Compute coherence
            frequencies, coherence = signal.coherence(
                s1, s2, 
                fs=self.sampling_rate,
                nperseg=min(256, len(s1)//4)
            )
            
            return {
                'frequencies': frequencies,
                'coherence': coherence,
                'mean_coherence': np.mean(coherence),
                'max_coherence': np.max(coherence),
                'max_coherence_freq': frequencies[np.argmax(coherence)]
            }
            
        except Exception as e:
            logger.error(f"Coherence analysis failed: {e}")
            return {}
    
    def plot_fft_analysis(self, fft_result: Dict, title: str = "FFT Analysis") -> plt.Figure:
        """Create FFT analysis plots."""
        if not fft_result:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        frequencies = fft_result['frequencies']
        magnitude = fft_result['magnitude']
        phase = fft_result['phase']
        psd = fft_result['psd']
        
        # Magnitude spectrum
        axes[0, 0].plot(frequencies, magnitude)
        axes[0, 0].set_title('Magnitude Spectrum')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase spectrum
        axes[0, 1].plot(frequencies, phase)
        axes[0, 1].set_title('Phase Spectrum')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Phase (rad)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Power Spectral Density
        axes[1, 0].semilogy(frequencies, psd)
        axes[1, 0].set_title('Power Spectral Density')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('PSD')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Dominant frequencies
        if len(fft_result['dominant_frequencies']) > 0:
            axes[1, 1].bar(range(len(fft_result['dominant_frequencies'])), 
                          fft_result['dominant_magnitudes'])
            axes[1, 1].set_title('Dominant Frequencies')
            axes[1, 1].set_xlabel('Rank')
            axes[1, 1].set_ylabel('Magnitude')
            axes[1, 1].set_xticks(range(len(fft_result['dominant_frequencies'])))
            axes[1, 1].set_xticklabels([f"{f:.1f} Hz" for f in fft_result['dominant_frequencies']], 
                                     rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_envelope_analysis(self, envelope_result: Dict, original_data: pd.Series, 
                             title: str = "Envelope Analysis") -> plt.Figure:
        """Create envelope analysis plots."""
        if not envelope_result:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Original signal vs envelope
        time_axis = np.arange(len(original_data))
        axes[0, 0].plot(time_axis, original_data.values, alpha=0.5, label='Original')
        axes[0, 0].plot(time_axis[:len(envelope_result['envelope'])], 
                       envelope_result['envelope'], 'r-', linewidth=2, label='Envelope')
        axes[0, 0].set_title('Signal and Envelope')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Envelope spectrum
        axes[0, 1].plot(envelope_result['envelope_frequencies'], 
                       envelope_result['envelope_magnitude'])
        axes[0, 1].set_title('Envelope Spectrum')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Envelope peaks
        if len(envelope_result['envelope_peaks_freq']) > 0:
            axes[1, 0].stem(envelope_result['envelope_peaks_freq'], 
                           envelope_result['envelope_peaks_mag'])
            axes[1, 0].set_title('Envelope Peaks')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Magnitude')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Envelope statistics
        axes[1, 1].text(0.1, 0.8, f"RMS: {envelope_result['envelope_rms']:.3f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Peak Count: {len(envelope_result['envelope_peaks_freq'])}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Envelope Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig