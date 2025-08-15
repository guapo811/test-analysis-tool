"""
Enhanced Error Handling Module
Provides comprehensive error handling, recovery mechanisms, and error reporting.
"""

import os
import sys
import traceback
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import functools

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, config: Dict):
        self.config = config['error_handling']
        self.error_log = []
        self.recovery_attempts = {}
        self.backup_folder = Path("error_backups")
        self.backup_folder.mkdir(exist_ok=True)
        
        # Setup detailed logging if enabled
        if self.config['detailed_logging']:
            self._setup_detailed_logging()
            
    def _setup_detailed_logging(self):
        """Setup detailed error logging."""
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler for errors
        error_log_file = Path("error_log.txt")
        file_handler = logging.FileHandler(error_log_file)
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(log_formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logger.info("Detailed error logging enabled")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None, 
                    recovery_function: Callable = None) -> Dict[str, Any]:
        """
        Handle errors with optional recovery and detailed reporting.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            recovery_function: Function to attempt recovery
            
        Returns:
            Dictionary containing error handling results
        """
        error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        error_info = {
            'error_id': error_id,
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'recovery_attempted': False,
            'recovery_successful': False,
            'status': 'logged'
        }
        
        # Log the error
        logger.error(f"Error {error_id}: {error_info['error_type']} - {error_info['error_message']}")
        
        if self.config['detailed_logging']:
            logger.error(f"Full traceback for {error_id}:\n{error_info['traceback']}")
            
        # Attempt recovery if enabled and function provided
        if self.config['enable_graceful_recovery'] and recovery_function:
            error_info['recovery_attempted'] = True
            
            try:
                recovery_result = self._attempt_recovery(error_id, recovery_function, context)
                error_info['recovery_successful'] = recovery_result['success']
                error_info['recovery_details'] = recovery_result
                
                if recovery_result['success']:
                    error_info['status'] = 'recovered'
                    logger.info(f"Successfully recovered from error {error_id}")
                else:
                    error_info['status'] = 'recovery_failed'
                    
            except Exception as recovery_error:
                error_info['recovery_error'] = str(recovery_error)
                error_info['status'] = 'recovery_failed'
                logger.error(f"Recovery failed for error {error_id}: {recovery_error}")
                
        # Store error information
        self.error_log.append(error_info)
        
        # Generate error report if enabled
        if self.config['error_report_generation']:
            self._generate_error_report(error_info)
            
        return error_info
    
    def _attempt_recovery(self, error_id: str, recovery_function: Callable, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt error recovery with retry logic."""
        max_attempts = self.config['auto_repair_attempts']
        
        if error_id not in self.recovery_attempts:
            self.recovery_attempts[error_id] = 0
            
        recovery_result = {
            'success': False,
            'attempts': 0,
            'max_attempts': max_attempts,
            'recovery_actions': []
        }
        
        for attempt in range(max_attempts):
            self.recovery_attempts[error_id] += 1
            recovery_result['attempts'] += 1
            
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{max_attempts} for error {error_id}")
                
                # Call the recovery function
                recovery_output = recovery_function(context)
                
                recovery_result['recovery_actions'].append({
                    'attempt': attempt + 1,
                    'action': 'recovery_function_called',
                    'result': 'success',
                    'output': str(recovery_output)
                })
                
                recovery_result['success'] = True
                break
                
            except Exception as recovery_error:
                recovery_result['recovery_actions'].append({
                    'attempt': attempt + 1,
                    'action': 'recovery_function_called',
                    'result': 'failed',
                    'error': str(recovery_error)
                })
                
                logger.warning(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
                
                # Wait before next attempt (simple backoff)
                if attempt < max_attempts - 1:
                    import time
                    time.sleep(1 * (attempt + 1))
                    
        return recovery_result
    
    def backup_corrupted_file(self, file_path: Path, error_context: str = "") -> Optional[Path]:
        """
        Backup a corrupted file for analysis.
        
        Args:
            file_path: Path to the corrupted file
            error_context: Description of the error context
            
        Returns:
            Path to the backup file if successful, None otherwise
        """
        if not self.config['backup_corrupted_files']:
            return None
            
        try:
            if not file_path.exists():
                logger.warning(f"Cannot backup non-existent file: {file_path}")
                return None
                
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}_corrupted{file_path.suffix}"
            backup_path = self.backup_folder / backup_name
            
            # Copy the file
            shutil.copy2(file_path, backup_path)
            
            # Create metadata file
            metadata = {
                'original_path': str(file_path),
                'backup_timestamp': datetime.now().isoformat(),
                'error_context': error_context,
                'file_size': file_path.stat().st_size,
                'backup_reason': 'corrupted_file'
            }
            
            metadata_path = backup_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Backed up corrupted file to: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup corrupted file {file_path}: {e}")
            return None
    
    def validate_file_integrity(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate file integrity and suggest repairs.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary containing validation results and repair suggestions
        """
        validation_result = {
            'file_path': str(file_path),
            'is_valid': True,
            'issues_found': [],
            'repair_suggestions': [],
            'file_info': {}
        }
        
        try:
            if not file_path.exists():
                validation_result['is_valid'] = False
                validation_result['issues_found'].append('File does not exist')
                validation_result['repair_suggestions'].append('Check file path and permissions')
                return validation_result
                
            # Basic file information
            file_stat = file_path.stat()
            validation_result['file_info'] = {
                'size_bytes': file_stat.st_size,
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'is_readable': os.access(file_path, os.R_OK)
            }
            
            # Check if file is readable
            if not validation_result['file_info']['is_readable']:
                validation_result['is_valid'] = False
                validation_result['issues_found'].append('File is not readable')
                validation_result['repair_suggestions'].append('Check file permissions')
                
            # Check file size
            if file_stat.st_size == 0:
                validation_result['is_valid'] = False
                validation_result['issues_found'].append('File is empty')
                validation_result['repair_suggestions'].append('File may be corrupted or incomplete')
                
            # CSV-specific validation
            if file_path.suffix.lower() == '.csv':
                csv_validation = self._validate_csv_file(file_path)
                validation_result.update(csv_validation)
                
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues_found'].append(f'Validation error: {str(e)}')
            validation_result['repair_suggestions'].append('File may be corrupted')
            
        return validation_result
    
    def _validate_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate CSV file specifically."""
        csv_validation = {
            'csv_issues': [],
            'csv_repairs': []
        }
        
        try:
            # Try to read the first few lines
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [f.readline() for _ in range(10)]
                
            # Check for common CSV issues
            if not lines[0].strip():
                csv_validation['csv_issues'].append('Empty first line')
                csv_validation['csv_repairs'].append('Remove empty lines from beginning')
                
            # Check for consistent delimiter
            if len(lines) > 1:
                delimiters = [',', ';', '\t', '|']
                delimiter_counts = {}
                
                for delimiter in delimiters:
                    counts = [line.count(delimiter) for line in lines[:5] if line.strip()]
                    if counts and len(set(counts)) == 1:  # Consistent count
                        delimiter_counts[delimiter] = counts[0]
                        
                if not delimiter_counts:
                    csv_validation['csv_issues'].append('No consistent delimiter found')
                    csv_validation['csv_repairs'].append('Check file format and delimiter')
                    
        except UnicodeDecodeError:
            csv_validation['csv_issues'].append('File encoding issue')
            csv_validation['csv_repairs'].append('Try different encoding (utf-8, latin-1, etc.)')
        except Exception as e:
            csv_validation['csv_issues'].append(f'CSV reading error: {str(e)}')
            
        return csv_validation
    
    def suggest_data_repair(self, data_issues: Dict[str, Any]) -> List[str]:
        """
        Suggest data repair strategies based on identified issues.
        
        Args:
            data_issues: Dictionary containing data quality issues
            
        Returns:
            List of repair suggestions
        """
        suggestions = []
        
        # Missing data suggestions
        if 'missing_data' in data_issues:
            missing_info = data_issues['missing_data']
            if missing_info.get('total_missing_cells', 0) > 0:
                suggestions.extend([
                    "Use interpolation to fill missing values",
                    "Remove rows with excessive missing data",
                    "Use forward/backward fill for time series data",
                    "Replace missing values with column mean/median"
                ])
                
        # Outlier suggestions
        if 'outlier_analysis' in data_issues:
            outlier_info = data_issues['outlier_analysis']
            if outlier_info.get('total_outliers', 0) > 0:
                suggestions.extend([
                    "Review outliers manually before removal",
                    "Use robust scaling to handle outliers",
                    "Apply winsorization to cap extreme values",
                    "Transform data using log or box-cox"
                ])
                
        # Data type suggestions
        if 'type_recommendations' in data_issues:
            suggestions.extend(data_issues['type_recommendations'])
            
        # Time series suggestions
        if 'time_warnings' in data_issues and data_issues['time_warnings']:
            suggestions.extend([
                "Sort data by timestamp",
                "Remove duplicate timestamps",
                "Check for timezone issues",
                "Resample data to regular intervals"
            ])
            
        return suggestions
    
    def _generate_error_report(self, error_info: Dict[str, Any]):
        """Generate detailed error report."""
        try:
            report_filename = f"error_report_{error_info['error_id']}.txt"
            report_path = Path("error_reports")
            report_path.mkdir(exist_ok=True)
            
            full_report_path = report_path / report_filename
            
            with open(full_report_path, 'w') as f:
                f.write("ERROR REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Error ID: {error_info['error_id']}\n")
                f.write(f"Timestamp: {error_info['timestamp']}\n")
                f.write(f"Error Type: {error_info['error_type']}\n")
                f.write(f"Error Message: {error_info['error_message']}\n\n")
                
                f.write("CONTEXT:\n")
                f.write("-" * 20 + "\n")
                for key, value in error_info['context'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("TRACEBACK:\n")
                f.write("-" * 20 + "\n")
                f.write(error_info['traceback'])
                f.write("\n")
                
                if error_info['recovery_attempted']:
                    f.write("RECOVERY ATTEMPT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Success: {error_info['recovery_successful']}\n")
                    if 'recovery_details' in error_info:
                        f.write(f"Details: {error_info['recovery_details']}\n")
                    f.write("\n")
                    
            logger.info(f"Error report generated: {full_report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate error report: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_log:
            return {'total_errors': 0, 'status': 'no_errors'}
            
        error_types = {}
        recovery_stats = {'attempted': 0, 'successful': 0}
        
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error['recovery_attempted']:
                recovery_stats['attempted'] += 1
                if error['recovery_successful']:
                    recovery_stats['successful'] += 1
                    
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recovery_statistics': recovery_stats,
            'most_recent_error': self.error_log[-1]['timestamp'],
            'error_rate': len(self.error_log) / max(1, len(self.error_log))  # Simple rate
        }
    
    def clear_error_log(self):
        """Clear the error log."""
        self.error_log.clear()
        self.recovery_attempts.clear()
        logger.info("Error log cleared")


def error_handler_decorator(error_handler: ErrorHandler, recovery_function: Callable = None):
    """
    Decorator to automatically handle errors with the ErrorHandler.
    
    Args:
        error_handler: ErrorHandler instance
        recovery_function: Optional recovery function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function_name': func.__name__,
                    'args': str(args)[:100],  # Limit length
                    'kwargs': str(kwargs)[:100]
                }
                
                error_info = error_handler.handle_error(e, context, recovery_function)
                
                # Decide whether to re-raise based on configuration
                if not error_handler.config['continue_on_error']:
                    raise
                    
                # Return None or appropriate default value
                return None
                
        return wrapper
    return decorator


class ContextualErrorHandler:
    """Context manager for handling errors in specific contexts."""
    
    def __init__(self, error_handler: ErrorHandler, context: Dict[str, Any], 
                 recovery_function: Callable = None):
        self.error_handler = error_handler
        self.context = context
        self.recovery_function = recovery_function
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.handle_error(exc_val, self.context, self.recovery_function)
            
            # Suppress exception if continue_on_error is enabled
            if self.error_handler.config['continue_on_error']:
                return True
                
        return False