# Enhanced BLDC Motor Data Analyzer

A comprehensive, advanced analysis tool for BLDC motor dynamometer data with integrated signal processing, predictive analytics, motor-specific analysis, and robust error handling.

## 🚀 Core Features

### Traditional Analysis
- **Automated CSV Analysis**: Intelligently identifies column types (vibration, speed, torque, temperature, etc.)
- **Comprehensive Vibration Analysis**: RMS, peak-to-peak, statistical analysis, and severity calculations
- **Professional Plotting**: Individual file analysis with multiple plot types
- **Multi-file Comparison**: Compare trends across multiple test sessions
- **Memory Efficient**: Designed to handle large datasets with chunked processing
- **Parallel Processing**: Multi-core support for analyzing multiple files
- **Detailed Reports**: Automated generation of analysis reports

### 🔬 Advanced Signal Processing
- **FFT Analysis**: Frequency domain analysis with configurable windowing
- **Envelope Analysis**: Hilbert transform-based envelope detection
- **Digital Filtering**: Configurable low-pass, high-pass, band-pass filters
- **Spectrogram Analysis**: Time-frequency analysis
- **Order Analysis**: RPM-based frequency analysis
- **Coherence Analysis**: Cross-correlation between signals

### 🛡️ Data Quality Management
- **Comprehensive Validation**: Automatic data integrity checks
- **Missing Data Handling**: Multiple interpolation methods
- **Outlier Detection**: IQR, Z-score, and modified Z-score methods
- **Data Cleaning**: Automated data repair and cleaning
- **Type Validation**: Automatic data type detection and conversion

### 🤖 Predictive Analytics
- **Anomaly Detection**: Isolation Forest, One-Class SVM, statistical methods
- **Trend Analysis**: Linear trend detection and change point analysis
- **Pattern Recognition**: Vibration pattern analysis and fault detection
- **Machine Learning**: Model training for future predictions
- **Seasonal Analysis**: Automatic seasonality detection

### ⚡ Motor-Specific Analysis
- **Efficiency Calculation**: Real-time efficiency monitoring
- **Torque Ripple Analysis**: Harmonic analysis and ripple characterization
- **Harmonic Analysis**: Current, voltage, and torque harmonics
- **Cogging Torque Detection**: Position-based and time-based analysis
- **Temperature Correlation**: Performance vs temperature analysis
- **Load Mapping**: Efficiency vs load characteristics

### 🚨 Enhanced Error Handling
- **Graceful Recovery**: Automatic error recovery with retry logic
- **File Validation**: Integrity checks before processing
- **Backup System**: Automatic backup of corrupted files
- **Detailed Logging**: Comprehensive error reporting
- **Contextual Errors**: Error handling with full context preservation

## 📁 Project Structure

```
CSVAnalysisPlotter/
├── motor_data_analyzer.py     # Main analysis engine (enhanced)
├── signal_processing.py       # Advanced signal processing module
├── data_quality.py           # Data quality management
├── predictive_analytics.py   # ML and predictive analysis
├── motor_analysis.py         # Motor-specific analysis
├── error_handling.py         # Enhanced error handling
├── config.json              # Comprehensive configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 📊 Output Examples

The enhanced tool generates comprehensive visualizations:

### Traditional Plots
- **Vibration Analysis**: Time series plots with RMS indicators and distribution histograms
- **Performance Overview**: Multi-metric dashboard showing speed, torque, power, temperature
- **Time Series Analysis**: Normalized comparison of multiple parameters over time
- **Correlation Matrix**: Heat maps showing relationships between variables
- **Trend Analysis**: Cross-file comparison showing changes over multiple test sessions

### Advanced Analysis Plots
- **FFT Analysis**: Frequency domain plots with magnitude, phase, and PSD
- **Envelope Analysis**: Signal envelope and envelope spectrum
- **Motor Efficiency**: Efficiency vs load with power comparison
- **Torque Ripple**: Harmonic content and ripple characteristics
- **Anomaly Detection**: Highlighted anomalies in time series data

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CSVAnalysisPlotter.git
cd CSVAnalysisPlotter
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

The enhanced analyzer uses a comprehensive JSON configuration file (`config.json`) with these key sections:

### Data Settings
```json
{
  "data_settings": {
    "data_folder": "./data",
    "chunk_size": 10000,
    "large_file_threshold_mb": 100
  }
}
```

### Signal Processing
```json
{
  "signal_processing": {
    "enable_fft_analysis": true,
    "fft_window": "hann",
    "enable_filtering": true,
    "enable_envelope_analysis": true
  }
}
```

### Predictive Analytics
```json
{
  "predictive_analytics": {
    "enable_anomaly_detection": true,
    "anomaly_methods": ["isolation_forest", "one_class_svm"],
    "enable_trend_analysis": true
  }
}
```

## 🚀 Usage

### Basic Usage
```bash
# Use default configuration
python motor_data_analyzer.py /path/to/data

# Use custom configuration
python motor_data_analyzer.py /path/to/data --config custom_config.json

# Limit files and use verbose logging
python motor_data_analyzer.py /path/to/data --max-files 10 --verbose
```

### Advanced Usage
```bash
# Sequential processing (disable multiprocessing)
python motor_data_analyzer.py /path/to/data --sequential

# Custom output folder
python motor_data_analyzer.py /path/to/data --output enhanced_results
```

### Command Line Options
```bash
python motor_data_analyzer.py [data_folder] [options]

Options:
  -o, --output OUTPUT_FOLDER    Output folder for results (default: analysis_results)
  -c, --config CONFIG_FILE      Configuration file path (default: config.json)
  -m, --max-files MAX_FILES     Maximum number of files to process
  --sequential                  Disable multiprocessing
  -v, --verbose                 Enable verbose logging
```

### Expected CSV Format

The tool automatically detects column types based on common naming patterns:

- **Time**: `time`, `timestamp`, `sec`, `seconds`
- **Vibration**: `vib`, `vibration`, `accel`, `acceleration`
- **Speed**: `speed`, `rpm`, `velocity`, `rotation`
- **Torque**: `torque`, `force`, `moment`
- **Power**: `power`, `watt`, `kw`
- **Temperature**: `temp`, `temperature`, `thermal`
- **Current**: `current`, `amp`, `ampere`
- **Voltage**: `voltage`, `volt`, `potential`

Example CSV structure:
```csv
time_seconds,speed_rpm,torque_nm,vibration_x_axis_g,vibration_y_axis_g,temperature_c
0.1,1000,5.2,0.05,0.03,25.5
0.2,1020,5.3,0.06,0.04,25.7
...
```

## Sample Data

The repository includes sample motor test data in the `sample_data/` directory:

- 6 test sessions with varying parameters
- Realistic BLDC motor measurements
- Progressive vibration increase across sessions
- Generated using `sample_data_generator.py`

To generate additional sample data:

```bash
python sample_data_generator.py
```

## 📂 Output Structure

The enhanced tool creates comprehensive organized output:

```
analysis_results/
├── individual_plots/
│   ├── filename_vibration_analysis.png      # Traditional vibration plots
│   ├── filename_performance_overview.png    # Performance dashboard
│   ├── filename_fft_analysis.png           # FFT frequency analysis
│   ├── filename_envelope_analysis.png      # Envelope analysis
│   ├── filename_efficiency_analysis.png    # Motor efficiency plots
│   ├── filename_torque_ripple.png         # Torque ripple analysis
│   └── filename_anomaly_detection.png      # Anomaly detection plots
├── comparison_plots/
│   ├── vibration_trends.png               # Multi-file vibration trends
│   └── file_statistics.png                # File statistics comparison
└── reports/
    ├── analysis_report.txt                 # Comprehensive analysis report
    ├── data_quality_summary.txt           # Data quality assessment
    └── error_reports/                      # Detailed error reports
        └── error_report_[timestamp].txt
```

## 📈 Key Metrics Calculated

### Traditional Vibration Analysis
- **RMS (Root Mean Square)**: Overall vibration energy
- **Peak-to-Peak**: Maximum vibration range
- **Statistical Analysis**: Mean, standard deviation, percentiles
- **Severity Index**: Coefficient of variation for vibration assessment

### Advanced Signal Processing
- **FFT Analysis**: Frequency domain characteristics, dominant frequencies, power spectral density
- **Envelope Analysis**: Bearing fault indicators, envelope spectrum, peak detection
- **Filter Analysis**: Signal conditioning and noise reduction
- **Order Analysis**: RPM-synchronized frequency analysis

### Motor-Specific Metrics
- **Efficiency Analysis**: Real-time efficiency, load mapping, power comparison
- **Torque Ripple**: Harmonic content, ripple factor, peak-to-peak ripple
- **Harmonic Analysis**: Current/voltage harmonics, THD calculation
- **Cogging Detection**: Position-dependent torque variations

### Predictive Analytics
- **Anomaly Scores**: Isolation forest, SVM, statistical outlier detection
- **Trend Analysis**: Linear trends, change points, statistical significance
- **Pattern Recognition**: Motor-specific pattern identification
- **Data Quality**: Validation scores, missing data analysis, outlier statistics

### Performance Metrics
- **Speed Analysis**: RPM trends and stability
- **Torque Characteristics**: Load analysis
- **Power Calculations**: Efficiency metrics
- **Temperature Monitoring**: Thermal performance and correlations

## API Usage

For programmatic use:

```python
from motor_data_analyzer import MotorDataAnalyzer
from pathlib import Path

# Initialize enhanced analyzer with custom config
analyzer = MotorDataAnalyzer(
    data_folder="path/to/csv", 
    output_folder="results",
    config_path="custom_config.json"
)

# Set motor parameters for specific analysis
analyzer.motor_analyzer.set_motor_parameters(
    pole_pairs=4,
    rated_voltage=24,
    rated_current=10
)

# Analyze single file with comprehensive analysis
stats = analyzer.analyze_single_file(Path("motor_test.csv"))

# Access specific analysis results
if 'efficiency_analysis' in stats:
    efficiency_data = stats['efficiency_analysis']
    print(f"Mean efficiency: {efficiency_data['efficiency_statistics']['mean_efficiency']:.2f}%")

if 'anomaly_detection' in stats:
    anomalies = stats['anomaly_detection']['summary']
    print(f"Found {anomalies['total_anomalies']} anomalies")

# Batch processing with error handling
csv_files = analyzer.get_csv_files()
all_stats = analyzer.process_batch(max_files=None, use_multiprocessing=True)

# Generate comparison analysis
analyzer.compare_files(all_stats)

# Access error handling results
error_summary = analyzer.error_handler.get_error_summary()
print(f"Processing completed with {error_summary['total_errors']} errors")
```

## Memory Optimization

For large datasets:

- **Chunked Processing**: Automatically handles large files
- **Memory Monitoring**: Efficient garbage collection
- **Selective Analysis**: Focus on specific columns or time ranges
- **Parallel Processing**: Distribute load across CPU cores

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Adjust chunk size in config
   ```json
   {"data_settings": {"chunk_size": 5000}}
   ```

3. **Processing Errors**: Enable detailed logging
   ```bash
   python motor_data_analyzer.py /path/to/data --verbose
   ```

4. **Configuration Issues**: Validate JSON syntax
   ```bash
   python -m json.tool config.json
   ```

5. **Column Recognition**: Check column naming conventions or update config keywords

6. **Permission Errors**: Check write permissions for output folder

### Debug Mode

Enable comprehensive logging and error reporting:

```bash
# Verbose logging with error details
python motor_data_analyzer.py sample_data --verbose

# Check configuration
python -c "import json; print(json.load(open('config.json')))"

# Test with single file
python motor_data_analyzer.py sample_data --max-files 1 --verbose
```

### Error Reports

The enhanced system automatically generates detailed error reports in `error_reports/` folder with:
- Full error context and traceback
- Recovery attempts and results
- Suggested repair actions
- File backup information

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis feature'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## 📦 Requirements

See `requirements.txt` for complete dependency list:

### Core Dependencies
- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computations
- **matplotlib** (≥3.5.0): Plotting and visualization
- **seaborn** (≥0.11.0): Statistical plotting enhancement

### Advanced Features
- **scipy** (≥1.8.0): Scientific computing and signal processing
- **scikit-learn** (≥1.1.0): Machine learning for predictive analytics
- **joblib** (≥1.1.0): Model persistence and parallel processing

### Installation
```bash
pip install -r requirements.txt
```

## 🎯 Feature Highlights

### What's New in Enhanced Version
- **40+ new analysis functions** across signal processing, predictive analytics, and motor analysis
- **Comprehensive configuration** through JSON config file
- **Robust error handling** with automatic recovery and detailed reporting
- **Advanced plotting** with FFT, envelope, efficiency, and anomaly visualizations
- **Machine learning integration** for anomaly detection and trend analysis
- **Motor expertise** with industry-standard analysis methods
- **Professional reporting** with detailed analysis summaries

### Backward Compatibility
The enhanced version maintains full backward compatibility. All original functionality works unchanged, with new features enabled through configuration options.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- Designed for BLDC motor dynamometer test analysis
- Optimized for vibration and performance monitoring
- Built with industrial data analysis best practices
- Enhanced with advanced signal processing and machine learning
- Comprehensive error handling for production environments

## 🆘 Support

For issues or questions:
1. Check error reports in `error_reports/` folder
2. Review log files for detailed information
3. Validate configuration file syntax
4. Ensure data files meet format requirements
5. Open an issue on GitHub with detailed error information

## 📞 Contact

For questions, issues, or contributions, please open an issue on GitHub with:
- Error logs and configuration details
- Sample data (if possible)
- System information and Python version
- Expected vs actual behavior description

---

**Enhanced BLDC Motor Data Analyzer** - From basic CSV analysis to comprehensive motor diagnostics with advanced signal processing, predictive analytics, and robust error handling.