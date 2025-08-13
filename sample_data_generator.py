#!/usr/bin/env python3
"""
Sample data generator for testing the motor data analyzer.
Creates realistic BLDC motor dynamometer test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_sample_data(filename: str, num_samples: int = 1000, test_duration_hours: float = 0):
    """Generate realistic motor test data with vibration and performance metrics."""
    
    # Time vector
    time = np.linspace(0, 10, num_samples)  # 10 seconds of data
    
    # Base motor operating conditions
    base_speed = 3000  # RPM
    base_torque = 5.0  # Nm
    base_power = base_speed * base_torque * 2 * np.pi / 60 / 1000  # kW approximation
    
    # Add some degradation based on test duration (simulating wear)
    degradation_factor = 1 + test_duration_hours * 0.001  # 0.1% increase per hour
    
    # Generate data with realistic noise and trends
    data = {
        'timestamp': time,
        'motor_speed_rpm': base_speed + np.sin(0.5 * time) * 100 + np.random.normal(0, 20, num_samples),
        'motor_torque_nm': base_torque + np.sin(0.3 * time) * 0.5 + np.random.normal(0, 0.1, num_samples),
        'motor_power_kw': base_power + np.sin(0.4 * time) * 0.2 + np.random.normal(0, 0.05, num_samples),
        'motor_current_a': 10 + np.sin(0.6 * time) * 1 + np.random.normal(0, 0.2, num_samples),
        'motor_voltage_v': 48 + np.sin(0.2 * time) * 2 + np.random.normal(0, 0.5, num_samples),
        'motor_temperature_c': 45 + time * 0.5 + np.random.normal(0, 2, num_samples),  # Heating up over time
        
        # Vibration data (most important for analysis)
        'vibration_x_axis_g': (0.5 + np.sin(10 * time) * 0.3 + np.sin(50 * time) * 0.1) * degradation_factor + np.random.normal(0, 0.05, num_samples),
        'vibration_y_axis_g': (0.4 + np.sin(12 * time) * 0.25 + np.sin(45 * time) * 0.08) * degradation_factor + np.random.normal(0, 0.04, num_samples),
        'vibration_z_axis_g': (0.3 + np.sin(8 * time) * 0.2 + np.sin(60 * time) * 0.06) * degradation_factor + np.random.normal(0, 0.03, num_samples),
        'vibration_overall_g': np.sqrt(
            (0.5 + np.sin(10 * time) * 0.3)**2 + 
            (0.4 + np.sin(12 * time) * 0.25)**2 + 
            (0.3 + np.sin(8 * time) * 0.2)**2
        ) * degradation_factor + np.random.normal(0, 0.08, num_samples),
        
        # Additional sensors
        'bearing_temperature_c': 35 + time * 0.3 + np.random.normal(0, 1.5, num_samples),
        'ambient_temperature_c': 22 + np.sin(0.1 * time) * 3 + np.random.normal(0, 1, num_samples),
        'supply_pressure_bar': 6.0 + np.sin(0.15 * time) * 0.2 + np.random.normal(0, 0.1, num_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic constraints and correlations
    df.loc[df['motor_speed_rpm'] < 0, 'motor_speed_rpm'] = 0  # No negative speed
    df.loc[df['motor_torque_nm'] < 0, 'motor_torque_nm'] = 0  # No negative torque
    df.loc[df['motor_temperature_c'] < 20, 'motor_temperature_c'] = 20  # Minimum temperature
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {num_samples} samples")
    
    return df

def create_test_dataset():
    """Create a set of test files simulating motor degradation over time."""
    
    # Create sample_data folder
    data_folder = Path("sample_data")
    data_folder.mkdir(exist_ok=True)
    
    # Generate files representing different test sessions over time
    test_sessions = [
        ("motor_test_session_001.csv", 1200, 0),      # Fresh motor
        ("motor_test_session_002.csv", 1150, 50),     # After 50 hours
        ("motor_test_session_003.csv", 1300, 100),    # After 100 hours
        ("motor_test_session_004.csv", 1400, 200),    # After 200 hours
        ("motor_test_session_005.csv", 1100, 350),    # After 350 hours
        ("motor_test_session_006.csv", 1250, 500),    # After 500 hours
    ]
    
    for filename, samples, hours in test_sessions:
        filepath = data_folder / filename
        generate_sample_data(str(filepath), samples, hours)
    
    print(f"\nTest dataset created in {data_folder}/")
    print(f"Generated {len(test_sessions)} files with varying degradation levels")
    print("\nTo analyze the data, run:")
    print(f"python motor_data_analyzer.py {data_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sample motor test data')
    parser.add_argument('--samples', '-n', type=int, default=1200, help='Number of samples per file')
    parser.add_argument('--files', '-f', type=int, default=6, help='Number of files to generate')
    
    args = parser.parse_args()
    
    create_test_dataset()