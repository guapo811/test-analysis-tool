# CSV Analysis Plotter

A comprehensive Python tool for analyzing and visualizing CSV data from BLDC motor dynamometer tests, with focus on vibration analysis and performance metrics.

## Features

- **Automated CSV Analysis**: Intelligently identifies column types (vibration, speed, torque, temperature, etc.)
- **Comprehensive Vibration Analysis**: RMS, peak-to-peak, statistical analysis, and severity calculations
- **Professional Plotting**: Individual file analysis with multiple plot types
- **Multi-file Comparison**: Compare trends across multiple test sessions
- **Memory Efficient**: Designed to handle large datasets with chunked processing
- **Parallel Processing**: Multi-core support for analyzing multiple files
- **Detailed Reports**: Automated generation of analysis reports

## Output Examples

The tool generates various types of visualizations:

- **Vibration Analysis**: Time series plots with RMS indicators and distribution histograms
- **Performance Overview**: Multi-metric dashboard showing speed, torque, power, temperature
- **Time Series Analysis**: Normalized comparison of multiple parameters over time
- **Correlation Matrix**: Heat maps showing relationships between variables
- **Trend Analysis**: Cross-file comparison showing changes over multiple test sessions

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

## Usage

### Basic Usage

Analyze CSV files in a directory:

```bash
python motor_data_analyzer.py path/to/csv/files
```

### Command Line Options

```bash
python motor_data_analyzer.py [data_folder] [options]

Options:
  -o, --output OUTPUT_FOLDER    Output folder for results (default: analysis_results)
  -p, --parallel               Enable parallel processing for multiple files
  -v, --verbose                Enable verbose logging
  --format FORMAT              Output format for plots (png, pdf, svg)
  --dpi DPI                    Resolution for output plots (default: 300)
```

### Example Commands

```bash
# Analyze sample data with default settings
python motor_data_analyzer.py sample_data

# Custom output folder and parallel processing
python motor_data_analyzer.py sample_data -o my_results -p

# High resolution PDF outputs
python motor_data_analyzer.py sample_data --format pdf --dpi 600
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

## Output Structure

The tool creates organized output folders:

```
analysis_results/
├── individual_plots/          # Individual file analysis
│   ├── session_001_vibration_analysis.png
│   ├── session_001_performance_overview.png
│   └── ...
├── comparison_plots/          # Multi-file comparisons
│   ├── vibration_trends.png
│   └── file_statistics.png
└── reports/                   # Text reports
    └── analysis_report.txt
```

## Key Metrics Calculated

### Vibration Analysis
- **RMS (Root Mean Square)**: Overall vibration energy
- **Peak-to-Peak**: Maximum vibration range
- **Statistical Analysis**: Mean, standard deviation, percentiles
- **Severity Index**: Coefficient of variation for vibration assessment

### Performance Metrics
- **Speed Analysis**: RPM trends and stability
- **Torque Characteristics**: Load analysis
- **Power Calculations**: Efficiency metrics
- **Temperature Monitoring**: Thermal performance

## API Usage

For programmatic use:

```python
from motor_data_analyzer import MotorDataAnalyzer

# Initialize analyzer
analyzer = MotorDataAnalyzer(data_folder="path/to/csv", output_folder="results")

# Analyze single file
stats = analyzer.analyze_single_file("motor_test.csv")

# Analyze all files in folder
csv_files = analyzer.get_csv_files()
all_stats = []
for file in csv_files:
    stats = analyzer.analyze_single_file(file)
    all_stats.append(stats)

# Generate comparison plots
analyzer.compare_files(all_stats)
```

## Memory Optimization

For large datasets:

- **Chunked Processing**: Automatically handles large files
- **Memory Monitoring**: Efficient garbage collection
- **Selective Analysis**: Focus on specific columns or time ranges
- **Parallel Processing**: Distribute load across CPU cores

## Troubleshooting

### Common Issues

1. **Memory Errors**: Use chunked processing for files >100MB
2. **Column Recognition**: Check column naming conventions
3. **Plot Display**: Ensure matplotlib backend is properly configured
4. **Permission Errors**: Check write permissions for output folder

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python motor_data_analyzer.py sample_data -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis feature'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## Requirements

See `requirements.txt` for complete dependency list:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical plotting enhancement
- **pathlib**: Path handling (Python 3.4+)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Designed for BLDC motor dynamometer test analysis
- Optimized for vibration and performance monitoring
- Built with industrial data analysis best practices

## Contact

For questions, issues, or contributions, please open an issue on GitHub.