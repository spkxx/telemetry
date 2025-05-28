"""Log and Telemetry Visualization Application

A Streamlit application for visualizing log and telemetry data with advanced filtering
and display options. Supports multiple file uploads, time normalization, and interactive filtering.

Features:
- Multi-file upload with color coding
- Time alignment for comparing data from different periods
- Interactive filtering by component, time, and text
- File-grouped visualization with custom legends
- Tabular data display with filtering options
"""

import streamlit as st
import pandas as pd
import re
import os
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LogTeleVisualizer:
    """
    A class to visualize log and telemetry data with advanced filtering and display options.
    Supports multiple file uploads, time normalization, and interactive filtering.
    """
    
    def __init__(self):
        """Initialize the visualizer with empty data structures and default settings."""
        # Data storage
        self.log_files = {}  # Dictionary to store multiple log DataFrames with their file names
        self.telemetry_files = {}  # Dictionary to store multiple telemetry DataFrames with their file names
        
        # Metadata
        self.components = []  # List of unique components from log files
        self.telemetry_columns = []  # List of numeric columns from telemetry files
        
        # Visual elements
        self.component_colors = {}  # Colors for components
        self.file_colors = {}  # Colors for files
        
        # Filtered data
        self.filtered_log_df = None  # Filtered log DataFrame based on current selections
        self.filtered_tele_df = None  # Filtered telemetry DataFrame based on current selections
        
        # User selections
        self.selected_components = []  # Selected log components
        self.selected_tele_columns = []  # Selected telemetry columns
        self.time_window = [None, None]  # Start and end time
        self.search_text = ""  # Text search in log messages
        self.view_mode = "combined"  # "log", "telemetry", or "combined"
        self.active_log_files = []  # List of active log file names
        self.active_telemetry_files = []  # List of active telemetry file names
        self.normalize_time = False  # Whether to normalize timestamps for overlay
        
        # Color palettes - using bright, vibrant colors that are easy to distinguish
        # Avoiding dark colors for better visibility
        self.log_color_palette = [
            # Bright vibrant colors
            '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#EF476F', '#FFC857', '#84BCDA',
            # Light but distinct colors
            '#F9C80E', '#F86624', '#EA3546', '#662E9B', '#43BCCD', '#C9E265', '#F9DB6D', '#FE5F55',
            # Pastel but clearly distinguishable
            '#FF9AA2', '#FFB7B2', '#FFDAC1', '#E2F0CB', '#B5EAD7', '#C7CEEA', '#F0E6EF', '#D4F0F0'
        ]
        
        # Use a different palette for telemetry to distinguish from logs
        self.tele_color_palette = [
            # Bright modern colors
            '#00BBF9', '#F15BB5', '#9B5DE5', '#00F5D4', '#FEE440', '#FB5607', '#80FFDB', '#FCBF49',
            # Medium brightness colors
            '#F72585', '#7209B7', '#3A0CA3', '#4361EE', '#4CC9F0', '#A8E6CE', '#DCEDC2', '#FFD3B5',
            # Distinct but not dark
            '#FFAAA6', '#FF8C94', '#A8E6CF', '#DCEDC2', '#FFD3B6', '#FFDDE1', '#BDCEBE', '#FFAAA5'
        ]
        
        # Ensure we have enough colors by duplicating if needed
        self.log_color_palette = (self.log_color_palette * 4)[:100]  # Up to 100 unique colors
        self.tele_color_palette = (self.tele_color_palette * 4)[:100]  # Up to 100 unique colors
        
    def parse_log(self, log_file):
        """Parse the log file into a pandas DataFrame with robust error handling.
        
        Args:
            log_file: Uploaded log file object from Streamlit
            
        Returns:
            DataFrame or None: Parsed log data or None if parsing failed
        """
        st.text(f"Parsing log file: {log_file.name}")
        try:
            # Set a maximum number of lines to process to prevent freezing on large files
            MAX_LINES = 10000
            data = []
            content = log_file.getvalue().decode('utf-8', errors='replace')
            lines = content.splitlines()[:MAX_LINES]
            
            # Show progress to user
            progress_bar = st.progress(0)
            total_lines = len(lines)
            
            # Try different log formats - first the specific format
            specific_format_count = 0
            generic_format_count = 0
            
            for line_num, line in enumerate(lines):
                # Update progress bar every 100 lines
                if line_num % 100 == 0:
                    progress_bar.progress(line_num / total_lines)
                
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # First try the specific format
                match = re.match(r'may (\d+) (\d+:\d+:\d+\.\d+) (\S+)\s+(.*)', line)
                if match:
                    specific_format_count += 1
                    day, time_str, component, message = match.groups()
                    # Create timestamp (assuming current year)
                    timestamp = f"2025-05-{day} {time_str}"
                    
                    # Check if message contains JSON-like content
                    json_data = None
                    json_match = re.search(r'\{.*\}', message)
                    if json_match:
                        try:
                            json_data = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass  # Not valid JSON
                    
                    data.append({
                        'timestamp': timestamp,
                        'component': component,
                        'message': message,
                        'line_num': line_num,
                        'json_data': json_data,
                        'source_file': log_file.name
                    })
                else:
                    # Try a more generic format - look for any timestamp-like pattern
                    generic_match = re.search(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)', line)
                    if generic_match:
                        generic_format_count += 1
                        timestamp = generic_match.group(1)
                        # Rest of the line is the message
                        message = line
                        # Use filename as component if we can't extract it
                        component = os.path.splitext(os.path.basename(log_file.name))[0]
                        
                        data.append({
                            'timestamp': timestamp,
                            'component': component,
                            'message': message,
                            'line_num': line_num,
                            'json_data': None,
                            'source_file': log_file.name
                        })
            
            # Clear progress bar
            progress_bar.empty()
            
            if not data:
                st.warning(f"Warning: No data parsed from log file {log_file.name}")
                st.info("The file may not match any supported log format.")
                return None
                
            df = pd.DataFrame(data)
            
            try:
                # Convert timestamp to datetime (ensure tz-naive)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # Drop rows with invalid timestamps
                invalid_timestamps = df['timestamp'].isna().sum()
                if invalid_timestamps > 0:
                    st.warning(f"Dropped {invalid_timestamps} rows with invalid timestamps")
                    df = df.dropna(subset=['timestamp'])
            except Exception as e:
                st.error(f"Error converting timestamps in {log_file.name}: {str(e)}")
                # Try to create a simple dataframe with just the messages to avoid complete failure
                try:
                    df = pd.DataFrame({
                        'timestamp': pd.Timestamp.now(),
                        'component': os.path.basename(log_file.name),
                        'message': [line for line in lines if line.strip()],
                        'source_file': log_file.name
                    })
                    st.warning("Created a simple dataframe with messages only due to timestamp conversion failure")
                except Exception as e:
                    st.error(f"Failed to create fallback dataframe: {str(e)}")
                    return None
            
            # Extract operation names if available - with better error handling
            try:
                df['operation'] = df['message'].apply(
                    lambda x: re.search(r'Operation start: (\w+)', str(x)).group(1) 
                    if x is not None and isinstance(x, str) and re.search(r'Operation start: (\w+)', x) else None
                )
            except Exception:
                df['operation'] = None
            
            # Extract status codes - with better error handling
            try:
                df['status'] = df['message'].apply(
                    lambda x: re.search(r'status (0x[0-9A-Fa-f]+)', str(x)).group(1) 
                    if x is not None and isinstance(x, str) and re.search(r'status (0x[0-9A-Fa-f]+)', x) else None
                )
            except Exception:
                df['status'] = None
            
            # Report parsing results
            if specific_format_count > 0 and generic_format_count > 0:
                st.info(f"Parsed {specific_format_count} lines with specific format and {generic_format_count} lines with generic format")
            elif specific_format_count > 0:
                st.success(f"Parsed {len(df)} log entries from {log_file.name} using specific format")
            else:
                st.info(f"Parsed {len(df)} log entries from {log_file.name} using generic format")
                
            return df
            
        except Exception as e:
            st.error(f"Error parsing log file {log_file.name}: {str(e)}")
            return None
    
    def parse_telemetry(self, telemetry_file):
        """Parse the telemetry CSV file into a pandas DataFrame with robust error handling.
        
        Args:
            telemetry_file: Uploaded telemetry file object from Streamlit
            
        Returns:
            DataFrame or None: Parsed telemetry data or None if parsing failed
        """
        st.text(f"Parsing telemetry file: {telemetry_file.name}")
        try:
            # Read CSV file
            df = pd.read_csv(telemetry_file)
            
            # Add source file information
            df['source_file'] = telemetry_file.name
            
            # Convert timestamp to datetime (convert to tz-naive if needed)
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            except Exception as e:
                st.error(f"Error processing timestamps in {telemetry_file.name}: {str(e)}")
                # Try alternative timestamp formats
                try:
                    st.info("Attempting alternative timestamp format...")
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
                except Exception as e2:
                    st.error(f"Could not parse timestamps: {str(e2)}")
                    return None
            
            # Get numeric columns (excluding sample count, timestamp, and activity_uuid)
            exclude_cols = ['timestamp', 'sample_count', 'activity_uuid', 'source_file']
            numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_cols:
                st.warning(f"No numeric columns found in {telemetry_file.name}")
                return None
            
            # Update telemetry columns list (union of all files)
            self.telemetry_columns = sorted(list(set(self.telemetry_columns + numeric_cols)))
            
            # Basic validation
            if len(df) == 0:
                st.warning(f"Telemetry file {telemetry_file.name} contains no data rows")
                return None
                
            # Check for missing values
            missing_count = df[numeric_cols].isna().sum().sum()
            if missing_count > 0:
                st.info(f"Found {missing_count} missing values in numeric columns. These will be handled during visualization.")
            
            st.success(f"Parsed telemetry file with {len(df)} rows and {len(numeric_cols)} numeric columns")
            return df
            
        except Exception as e:
            st.error(f"Error parsing telemetry file {telemetry_file.name}: {str(e)}")
            return None
            
    def apply_filters(self):
        """Apply component, time, and text filters to the data.
        
        This method filters the log and telemetry data based on:
        - Selected components (for logs)
        - Selected telemetry columns
        - Time window
        - Text search
        - Active files
        """
        # Initialize empty DataFrames
        self.filtered_log_df = pd.DataFrame()
        self.filtered_tele_df = pd.DataFrame()
        
        # Process log files if in log or combined mode
        if self.view_mode in ["log", "combined"] and self.active_log_files:
            # Combine active log files
            log_dfs = []
            for file_name in self.active_log_files:
                if file_name in self.log_files and self.log_files[file_name] is not None:
                    log_dfs.append(self.log_files[file_name])
            
            if log_dfs:
                # Concatenate all active log DataFrames
                try:
                    combined_log_df = pd.concat(log_dfs, ignore_index=True)
                    
                    # Apply time window filter if specified
                    if self.time_window[0] is not None and self.time_window[1] is not None:
                        combined_log_df = combined_log_df[
                            (combined_log_df['timestamp'] >= self.time_window[0]) &
                            (combined_log_df['timestamp'] <= self.time_window[1])
                        ]
                    
                    # Apply component filter if specified
                    if self.selected_components:
                        combined_log_df = combined_log_df[combined_log_df['component'].isin(self.selected_components)]
                    
                    # Apply text search if specified
                    if self.search_text:
                        combined_log_df = combined_log_df[combined_log_df['message'].str.contains(self.search_text, case=False, na=False)]
                    
                    # Sort by timestamp
                    combined_log_df = combined_log_df.sort_values('timestamp')
                    
                    # Update filtered DataFrame
                    self.filtered_log_df = combined_log_df
                except Exception as e:
                    st.error(f"Error applying log filters: {str(e)}")
        
        # Process telemetry files if in telemetry or combined mode
        if self.view_mode in ["telemetry", "combined"] and self.active_telemetry_files:
            # Combine active telemetry files
            tele_dfs = []
            for file_name in self.active_telemetry_files:
                if file_name in self.telemetry_files and self.telemetry_files[file_name] is not None:
                    tele_dfs.append(self.telemetry_files[file_name])
            
            if tele_dfs:
                # Concatenate all active telemetry DataFrames
                try:
                    combined_tele_df = pd.concat(tele_dfs, ignore_index=True)
                    
                    # Apply time window filter if specified
                    if self.time_window[0] is not None and self.time_window[1] is not None:
                        combined_tele_df = combined_tele_df[
                            (combined_tele_df['timestamp'] >= self.time_window[0]) &
                            (combined_tele_df['timestamp'] <= self.time_window[1])
                        ]
                    
                    # Sort by timestamp
                    combined_tele_df = combined_tele_df.sort_values('timestamp')
                    
                    # Update filtered DataFrame
                    self.filtered_tele_df = combined_tele_df
                except Exception as e:
                    st.error(f"Error applying telemetry filters: {str(e)}")
    
    def _normalize_log_timestamps(self):
        """Normalize timestamps for log data to overlay graphs from different time periods.
        
        This method adds a new column 'original_timestamp' with the original values,
        and shifts all timestamps to start at the same reference point.
        """
        if not self.normalize_time or self.filtered_log_df.empty:
            return
        
        try:
            # Store original timestamps
            self.filtered_log_df['original_timestamp'] = self.filtered_log_df['timestamp']
            
            # Group by source file
            file_groups = self.filtered_log_df.groupby('source_file')
            
            # Find the earliest timestamp across all files to use as reference
            global_min_time = self.filtered_log_df['timestamp'].min()
            
            # Process each file group
            normalized_dfs = []
            for file_name, group_df in file_groups:
                # Find the minimum timestamp for this file
                file_min_time = group_df['timestamp'].min()
                
                # Calculate the time difference to shift by
                time_diff = file_min_time - global_min_time
                
                # Create a copy of the group DataFrame
                normalized_df = group_df.copy()
                
                # Shift timestamps to align with the global minimum
                normalized_df['timestamp'] = normalized_df['timestamp'] - time_diff
                
                normalized_dfs.append(normalized_df)
            
            # Combine all normalized DataFrames
            if normalized_dfs:
                self.filtered_log_df = pd.concat(normalized_dfs, ignore_index=True)
                self.filtered_log_df = self.filtered_log_df.sort_values('timestamp')
        
        except Exception as e:
            st.error(f"Error normalizing log timestamps: {str(e)}")
    
    def _normalize_telemetry_timestamps(self):
        """Normalize timestamps for telemetry data to overlay graphs from different time periods.
        
        This method adds a new column 'original_timestamp' with the original values,
        and shifts all timestamps to start at the same reference point.
        """
        if not self.normalize_time or self.filtered_tele_df.empty:
            return
        
        try:
            # Store original timestamps
            self.filtered_tele_df['original_timestamp'] = self.filtered_tele_df['timestamp']
            
            # Group by source file
            file_groups = self.filtered_tele_df.groupby('source_file')
            
            # Find the earliest timestamp across all files to use as reference
            global_min_time = self.filtered_tele_df['timestamp'].min()
            
            # Process each file group
            normalized_dfs = []
            for file_name, group_df in file_groups:
                # Find the minimum timestamp for this file
                file_min_time = group_df['timestamp'].min()
                
                # Calculate the time difference to shift by
                time_diff = file_min_time - global_min_time
                
                # Create a copy of the group DataFrame
                normalized_df = group_df.copy()
                
                # Shift timestamps to align with the global minimum
                normalized_df['timestamp'] = normalized_df['timestamp'] - time_diff
                
                normalized_dfs.append(normalized_df)
            
            # Combine all normalized DataFrames
            if normalized_dfs:
                self.filtered_tele_df = pd.concat(normalized_dfs, ignore_index=True)
                self.filtered_tele_df = self.filtered_tele_df.sort_values('timestamp')
        
        except Exception as e:
            st.error(f"Error normalizing telemetry timestamps: {str(e)}")

# Helper functions for the main application
def toggle_log_file(visualizer, file_name):
    """Toggle a log file's active status."""
    if file_name in visualizer.active_log_files:
        visualizer.active_log_files.remove(file_name)
    else:
        visualizer.active_log_files.append(file_name)
    return True  # Indicate filter changed

def toggle_telemetry_file(visualizer, file_name):
    """Toggle a telemetry file's active status."""
    if file_name in visualizer.active_telemetry_files:
        visualizer.active_telemetry_files.remove(file_name)
    else:
        visualizer.active_telemetry_files.append(file_name)
    return True  # Indicate filter changed

# Main application function
def run_streamlit_app():
    """Main function to run the Streamlit application."""
    # Configure the page
    st.set_page_config(page_title="Log & Telemetry Visualizer", layout="wide")
    st.title("Log & Telemetry")
    
    # Initialize session state to prevent rerendering
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = LogTeleVisualizer()
        st.session_state.filter_changed = False
        st.session_state.last_active_log_files = []
        st.session_state.last_active_telemetry_files = []
    
    # Use session state visualizer
    visualizer = st.session_state.visualizer
    
    # Sidebar for file uploads and controls
    with st.sidebar:
        st.header("Data Sources")
        
        # Multiple log file upload
        log_files = st.file_uploader("Upload Log Files", type=["log", "txt"], accept_multiple_files=True)
        
        # Process log files
        if log_files:
            for log_file in log_files:
                if log_file.name not in visualizer.log_files:
                    parsed_df = visualizer.parse_log(log_file)
                    
                    if parsed_df is not None:
                        visualizer.log_files[log_file.name] = parsed_df
                        # Add to active files by default
                        if log_file.name not in visualizer.active_log_files:
                            visualizer.active_log_files.append(log_file.name)
                        
                        # Update components list (union of all files)
                        file_components = parsed_df['component'].unique().tolist()
                        visualizer.components = list(set(visualizer.components + file_components))
                        
                        # Initialize selected components if empty
                        if not visualizer.selected_components:
                            visualizer.selected_components = visualizer.components.copy()
        
        # Multiple telemetry file upload
        telemetry_files = st.file_uploader("Upload Telemetry Files", type=["csv"], accept_multiple_files=True)
        
        # Process telemetry files
        if telemetry_files:
            for telemetry_file in telemetry_files:
                if telemetry_file.name not in visualizer.telemetry_files:
                    parsed_df = visualizer.parse_telemetry(telemetry_file)
                    
                    if parsed_df is not None:
                        visualizer.telemetry_files[telemetry_file.name] = parsed_df
                        # Add to active files by default
                        if telemetry_file.name not in visualizer.active_telemetry_files:
                            visualizer.active_telemetry_files.append(telemetry_file.name)
                        
                        # Initialize selected telemetry columns if empty
                        if not visualizer.selected_tele_columns and visualizer.telemetry_columns:
                            # Select first 5 columns by default, or all if less than 5
                            visualizer.selected_tele_columns = visualizer.telemetry_columns[:5] if len(visualizer.telemetry_columns) > 5 else visualizer.telemetry_columns
        
        # Only show controls if data is loaded
        if visualizer.log_files or visualizer.telemetry_files:
            st.sidebar.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(100,100,100,0.3);'>", unsafe_allow_html=True)
            st.header("View Controls")
            
            # Time normalization option
            if (len(visualizer.log_files) > 1 or len(visualizer.telemetry_files) > 1):
                st.subheader("Time Alignment")
                new_normalize_setting = st.checkbox(
                    "Normalize timestamps (overlay graphs)",
                    value=visualizer.normalize_time,
                    help="When enabled, all files will have their timestamps normalized to start at the same point, allowing you to overlay and compare patterns regardless of when they occurred.",
                    key="time_normalization_toggle"
                )
                
                # If the setting changed, update and rerun
                if new_normalize_setting != visualizer.normalize_time:
                    visualizer.normalize_time = new_normalize_setting
                    st.session_state.filter_changed = True
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for older Streamlit versions
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.warning("Please refresh the page to apply time normalization changes.")
                
                if visualizer.normalize_time:
                    st.info("Time normalization is active. Timestamps shown are relative, not absolute.")
            
            # File toggles with color indicators
            if visualizer.log_files:
                st.sidebar.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(100,100,100,0.3);'>", unsafe_allow_html=True)
                st.subheader("Log Files")
                
                # Add Select All / Clear All buttons
                log_cols = st.columns(2)
                if log_cols[0].button("Select All Log Files"):
                    visualizer.active_log_files = list(visualizer.log_files.keys())
                    st.session_state.filter_changed = True
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for older Streamlit versions
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.warning("Could not refresh the page automatically. Please refresh manually.")
                if log_cols[1].button("Clear All Log Files"):
                    visualizer.active_log_files = []
                    st.session_state.filter_changed = True
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for older Streamlit versions
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.warning("Could not refresh the page automatically. Please refresh manually.")
                
                # Display file toggles with color indicators
                for file_name in visualizer.log_files.keys():
                    # Ensure file has a color assigned
                    if file_name not in visualizer.file_colors:
                        idx = len(visualizer.file_colors) % len(visualizer.log_color_palette)
                        visualizer.file_colors[file_name] = visualizer.log_color_palette[idx]
                    
                    # Create a colored checkbox
                    is_active = file_name in visualizer.active_log_files
                    file_color = visualizer.file_colors[file_name]
                    
                    # Use columns to show color indicator and checkbox
                    cols = st.columns([0.1, 0.9])
                    cols[0].markdown(f"<div style='background-color:{file_color}; width:20px; height:20px; border-radius:50%;'></div>", unsafe_allow_html=True)
                    
                    # Use a key that includes the file name to ensure uniqueness
                    if cols[1].checkbox(f"{file_name}", value=is_active, key=f"log_{file_name}"):
                        if not is_active:
                            visualizer.active_log_files.append(file_name)
                            st.session_state.filter_changed = True
                    elif is_active:  # Was checked but now unchecked
                        visualizer.active_log_files.remove(file_name)
                        st.session_state.filter_changed = True
            
            if visualizer.telemetry_files:
                st.sidebar.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(100,100,100,0.3);'>", unsafe_allow_html=True)
                st.subheader("Telemetry Files")
                
                # Add Select All / Clear All buttons
                tele_cols = st.columns(2)
                if tele_cols[0].button("Select All Telemetry Files"):
                    visualizer.active_telemetry_files = list(visualizer.telemetry_files.keys())
                    st.session_state.filter_changed = True
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for older Streamlit versions
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.warning("Could not refresh the page automatically. Please refresh manually.")
                if tele_cols[1].button("Clear All Telemetry Files"):
                    visualizer.active_telemetry_files = []
                    st.session_state.filter_changed = True
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback for older Streamlit versions
                        try:
                            st.experimental_rerun()
                        except Exception:
                            st.warning("Could not refresh the page automatically. Please refresh manually.")
                
                # Display file toggles with color indicators
                for file_name in visualizer.telemetry_files.keys():
                    # Ensure file has a color assigned
                    if file_name not in visualizer.file_colors:
                        idx = len(visualizer.file_colors) % len(visualizer.tele_color_palette)
                        visualizer.file_colors[file_name] = visualizer.tele_color_palette[idx]
                    
                    # Create a colored checkbox
                    is_active = file_name in visualizer.active_telemetry_files
                    file_color = visualizer.file_colors[file_name]
                    
                    # Use columns to show color indicator and checkbox
                    cols = st.columns([0.1, 0.9])
                    cols[0].markdown(f"<div style='background-color:{file_color}; width:20px; height:20px; border-radius:50%;'></div>", unsafe_allow_html=True)
                    
                    # Use a key that includes the file name to ensure uniqueness
                    if cols[1].checkbox(f"{file_name}", value=is_active, key=f"tele_{file_name}"):
                        if not is_active:
                            visualizer.active_telemetry_files.append(file_name)
                            st.session_state.filter_changed = True
                    elif is_active:  # Was checked but now unchecked
                        visualizer.active_telemetry_files.remove(file_name)
                        st.session_state.filter_changed = True
            
            # View mode selection
            st.sidebar.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(100,100,100,0.3);'>", unsafe_allow_html=True)
            visualizer.view_mode = st.radio(
                "View Mode",
                options=["combined", "log", "telemetry"],
                format_func=lambda x: x.capitalize()
            )
            
            # Time window controls
            st.sidebar.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(100,100,100,0.3);'>", unsafe_allow_html=True)
            st.subheader("Time Window")
            
            # Determine the time range from available data
            min_time = None
            max_time = None
            
            # Check all active log files
            for file_name in visualizer.active_log_files:
                if file_name in visualizer.log_files and visualizer.log_files[file_name] is not None:
                    log_df = visualizer.log_files[file_name]
                    log_min = log_df['timestamp'].min()
                    log_max = log_df['timestamp'].max()
                    min_time = log_min if min_time is None else min(min_time, log_min)
                    max_time = log_max if max_time is None else max(max_time, log_max)
            
            # Check all active telemetry files
            for file_name in visualizer.active_telemetry_files:
                if file_name in visualizer.telemetry_files and visualizer.telemetry_files[file_name] is not None:
                    tele_df = visualizer.telemetry_files[file_name]
                    tele_min = tele_df['timestamp'].min()
                    tele_max = tele_df['timestamp'].max()
                    min_time = tele_min if min_time is None else min(min_time, tele_min)
                    max_time = tele_max if max_time is None else max(max_time, tele_max)
            
            if min_time is not None and max_time is not None:
                # Convert to datetime for the date_input widget
                min_date = min_time.date()
                max_date = max_time.date()
                
                # Date selection
                selected_dates = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Time selection (hours and minutes)
                if len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    
                    # Get min/max times for the selected dates
                    start_time = st.slider(
                        "Start Time",
                        min_value=datetime.combine(start_date, datetime.min.time()),
                        max_value=datetime.combine(start_date, datetime.max.time()),
                        value=datetime.combine(start_date, datetime.min.time()),
                        format="HH:mm:ss"
                    )
                    
                    end_time = st.slider(
                        "End Time",
                        min_value=datetime.combine(end_date, datetime.min.time()),
                        max_value=datetime.combine(end_date, datetime.max.time()),
                        value=datetime.combine(end_date, datetime.max.time()),
                        format="HH:mm:ss"
                    )
                    
                    # Set the time window
                    visualizer.time_window = [start_time, end_time]
            
            # Text search
            visualizer.search_text = st.text_input("Search in Messages", value=visualizer.search_text)
            
            # Component selection
            if visualizer.components:
                st.sidebar.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(100,100,100,0.3);'>", unsafe_allow_html=True)
                st.subheader("Log Components")
                all_none_cols = st.columns(2)
                if all_none_cols[0].button("Select All Components"):
                    visualizer.selected_components = visualizer.components.copy()
                    st.session_state.filter_changed = True
                if all_none_cols[1].button("Clear Component Selection"):
                    visualizer.selected_components = []
                    st.session_state.filter_changed = True
                
                visualizer.selected_components = st.multiselect(
                    "Select Components",
                    options=sorted(visualizer.components),
                    default=visualizer.selected_components
                )
            
            # Telemetry column selection
            if visualizer.telemetry_columns:
                st.sidebar.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(100,100,100,0.3);'>", unsafe_allow_html=True)
                st.subheader("Telemetry Columns")
                all_none_cols = st.columns(2)
                if all_none_cols[0].button("Select All Telemetry"):
                    visualizer.selected_tele_columns = visualizer.telemetry_columns.copy()
                    st.session_state.filter_changed = True
                if all_none_cols[1].button("Clear Telemetry Selection"):
                    visualizer.selected_tele_columns = []
                    st.session_state.filter_changed = True
                
                # Quick selection buttons - arranged in a 3-column layout
                quick_cols = st.columns(3)
                if quick_cols[0].button("Temperature Columns"):
                    visualizer.selected_tele_columns = [col for col in visualizer.telemetry_columns if 'temperature' in col.lower() or 'temp' in col.lower()]
                    st.session_state.filter_changed = True
                if quick_cols[1].button("Step Count Columns"):
                    # Comprehensive filter for step count/motor columns
                    visualizer.selected_tele_columns = [col for col in visualizer.telemetry_columns 
                                                      if any(term in col.lower() for term in ['step', 'motor', 'count', 'position', 'move'])]
                    st.session_state.filter_changed = True
                if quick_cols[2].button("EWOD Columns"):
                    visualizer.selected_tele_columns = [col for col in visualizer.telemetry_columns if 'ewod' in col.lower()]
                    st.session_state.filter_changed = True
                
                visualizer.selected_tele_columns = st.multiselect(
                    "Select Telemetry Columns",
                    options=sorted(visualizer.telemetry_columns),
                    default=visualizer.selected_tele_columns
                )
    
    # Check if file selection has changed
    files_changed = (sorted(visualizer.active_log_files) != sorted(st.session_state.last_active_log_files) or
                     sorted(visualizer.active_telemetry_files) != sorted(st.session_state.last_active_telemetry_files))
    
    # Apply filters only if needed (when filters change or files change)
    if st.session_state.filter_changed or files_changed:
        visualizer.apply_filters()
        st.session_state.filter_changed = False
        st.session_state.last_active_log_files = visualizer.active_log_files.copy()
        st.session_state.last_active_telemetry_files = visualizer.active_telemetry_files.copy()
    
    # Main content area
    if not visualizer.log_files and not visualizer.telemetry_files:
        st.info("Please upload log and/or telemetry files to begin")
        return
    
    # Apply normalization if enabled
    if visualizer.normalize_time:
        visualizer._normalize_log_timestamps()
        visualizer._normalize_telemetry_timestamps()
    
    # Display a legend for file colors
    if visualizer.file_colors:
        st.sidebar.subheader("File Color Legend")
        for file_name, color in visualizer.file_colors.items():
            if ((file_name in visualizer.active_log_files) or 
                (file_name in visualizer.active_telemetry_files)):
                st.sidebar.markdown(f"<div style='display:flex;align-items:center;'>"
                                  f"<div style='background-color:{color}; width:15px; height:15px; "
                                  f"border-radius:50%; margin-right:10px;'></div>"
                                  f"<span style='font-size:0.8em;'>{file_name}</span></div>", 
                                  unsafe_allow_html=True)
    
    # Display visualizations based on view mode
    if visualizer.view_mode in ["log", "combined"] and not visualizer.filtered_log_df.empty and visualizer.selected_components:
        st.header("Log Timeline")
        
        # Create a figure for log events
        fig = go.Figure()
        
        # Create a dictionary to store traces by source file for toggling
        file_traces = {}
        
        # Group by source file and component
        for source_file in visualizer.filtered_log_df['source_file'].unique():
            file_df = visualizer.filtered_log_df[visualizer.filtered_log_df['source_file'] == source_file]
            file_components = file_df[file_df['component'].isin(visualizer.selected_components)]
            
            if len(file_components) == 0:
                continue
                
            # Get the file color and create variations for each component
            base_color = visualizer.file_colors.get(source_file, '#888888')
            
            # Add a trace for each component in this file
            for component in sorted(file_components['component'].unique()):
                component_df = file_components[file_components['component'] == component]
                
                # Generate a unique color for this component by adjusting the base color
                color_idx = (visualizer.components.index(component) if component in visualizer.components else 0) * 30
                if base_color.startswith('#'):
                    r, g, b = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
                    r = max(0, min(255, r + color_idx % 50 - 25))
                    g = max(0, min(255, g + (color_idx * 2) % 50 - 25))
                    b = max(0, min(255, b + (color_idx * 3) % 50 - 25))
                    marker_color = f'rgb({r},{g},{b})'
                else:
                    marker_color = base_color
                
                trace = go.Scatter(
                    x=component_df['timestamp'],
                    y=[component] * len(component_df),
                    mode='markers',
                    name=f"{component}",
                    legendgroup=source_file,  # Group by source file
                    legendgrouptitle_text=source_file,  # Show source file as group title
                    marker=dict(
                        color=marker_color,
                        size=10,
                        line=dict(width=1, color='DarkSlateGrey'),
                        opacity=0.8
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Time: %{x}<br>"
                        "File: " + source_file + "<br>"
                        "<extra></extra>"
                    )
                )
                fig.add_trace(trace)
        
        fig.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Component",
            hovermode="closest",
            legend=dict(
                orientation="v",  # Vertical legend
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                traceorder="grouped",  # Group by legendgroup
                groupclick="toggleitem"  # Click on group title toggles all traces in group
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show log entries in a table
        st.subheader("Log Entries")
        
        # Determine which columns to display based on whether time normalization is active
        display_columns = ['timestamp', 'component', 'message', 'status', 'operation', 'source_file']
        if visualizer.normalize_time and 'original_timestamp' in visualizer.filtered_log_df.columns:
            display_columns = ['timestamp', 'original_timestamp', 'component', 'message', 'status', 'operation', 'source_file']
        
        # Filter columns to only include those that exist
        available_cols = [col for col in display_columns if col in visualizer.filtered_log_df.columns]
        
        st.dataframe(
            visualizer.filtered_log_df[available_cols],
            use_container_width=True,
            height=300
        )
    elif visualizer.view_mode in ["log", "combined"]:
        st.warning("No log entries match the current filters")
    
    # Add divider between log and telemetry sections in combined view
    if visualizer.view_mode == "combined" and not visualizer.filtered_log_df.empty:
        st.markdown("<hr style='margin: 30px 0; border: 0; border-top: 2px solid rgba(100,100,100,0.2);'>", unsafe_allow_html=True)
    
    if visualizer.view_mode in ["telemetry", "combined"] and not visualizer.filtered_tele_df.empty and visualizer.selected_tele_columns:
        st.header("Telemetry Data")
        
        # Create a line plot for telemetry data
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create a dictionary to store traces by source file
        file_traces = {}
        
        # Group by source file and column, sort by column name for consistency
        for source_file in visualizer.filtered_tele_df['source_file'].unique():
            file_df = visualizer.filtered_tele_df[visualizer.filtered_tele_df['source_file'] == source_file]
            file_traces[source_file] = []
            
            # Get the file color
            file_color = visualizer.file_colors.get(source_file, '#888888')
            
            # Add traces for each selected column in this file
            for column in sorted(visualizer.selected_tele_columns):
                if column in file_df.columns:
                    # Create a color variation based on the column index
                    col_idx = visualizer.telemetry_columns.index(column) if column in visualizer.telemetry_columns else 0
                    
                    # Generate a color variation
                    if file_color.startswith('#'):
                        r, g, b = tuple(int(file_color[i:i+2], 16) for i in (1, 3, 5))
                        r = max(0, min(255, r + col_idx % 30 - 15))
                        g = max(0, min(255, g + (col_idx * 2) % 30 - 15))
                        b = max(0, min(255, b + (col_idx * 3) % 30 - 15))
                        line_color = f'rgb({r},{g},{b})'
                    else:
                        line_color = file_color
                    
                    trace = go.Scatter(
                        x=file_df["timestamp"],
                        y=file_df[column],
                        name=column,
                        legendgroup=source_file,  # Group by source file
                        legendgrouptitle_text=source_file,  # Show source file as group title
                        mode="lines",
                        line=dict(width=2, color=line_color),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Time: %{x}<br>"
                            "Column: " + column + "<br>"
                            "File: " + source_file + "<br>"
                            "<extra></extra>"
                        )
                    )
                    
                    # Store the trace for the custom legend
                    file_traces[source_file].append((column, line_color))
                    
                    # Add the trace to the figure
                    fig.add_trace(trace, secondary_y=False)
        
        fig.update_layout(
            height=500,
            title="Telemetry Data",
            xaxis_title="Time",
            hovermode="x unified",
            legend=dict(
                orientation="v",  # Vertical legend
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                traceorder="grouped",  # Group by legendgroup
                groupclick="toggleitem"  # Click on group title toggles all traces in group
            )
        )
        
        fig.update_yaxes(title_text="Value", secondary_y=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display custom file-based legend if there are multiple files
        if len(file_traces) > 1:
            st.subheader("Data Files Legend")
            cols = st.columns(min(len(file_traces), 3))  # Up to 3 columns
            
            for i, (source_file, traces) in enumerate(file_traces.items()):
                col_idx = i % len(cols)
                file_color = visualizer.file_colors.get(source_file, '#888888')
                
                # Create a styled box for each file
                cols[col_idx].markdown(
                    f'<div style="padding:10px; margin-bottom:10px; background-color:{file_color}20; '
                    f'border-left:4px solid {file_color}; border-radius:4px;">'
                    f'<b>{source_file}</b><br>'
                    f'<span style="font-size:0.8em;">Columns: {len(traces)}</span><br>'
                    f'<span style="font-size:0.8em; color:#666;">'
                    f'{", ".join([col for col, _ in traces[:3]])}'
                    f'{", ..." if len(traces) > 3 else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        # Show telemetry data in a table
        st.subheader("Telemetry Data Table")
        
        # Determine which columns to display based on whether time normalization is active
        base_cols = ["timestamp", "source_file"]
        if visualizer.normalize_time and 'original_timestamp' in visualizer.filtered_tele_df.columns:
            base_cols = ["timestamp", "original_timestamp", "source_file"]
            
        display_cols = base_cols + visualizer.selected_tele_columns
        
        # Filter to only include columns that exist
        available_cols = [col for col in display_cols if col in visualizer.filtered_tele_df.columns]
        
        # Add a file filter for the table
        if len(visualizer.filtered_tele_df['source_file'].unique()) > 1:
            selected_file_for_table = st.selectbox(
                "Filter table by file:",
                options=["All Files"] + list(visualizer.filtered_tele_df['source_file'].unique()),
                key="telemetry_table_file_filter"
            )
            
            if selected_file_for_table != "All Files":
                filtered_table_df = visualizer.filtered_tele_df[visualizer.filtered_tele_df['source_file'] == selected_file_for_table]
            else:
                filtered_table_df = visualizer.filtered_tele_df
        else:
            filtered_table_df = visualizer.filtered_tele_df
        
        st.dataframe(
            filtered_table_df[available_cols],
            use_container_width=True,
            height=300
        )
    elif visualizer.view_mode in ["telemetry", "combined"]:
        if not visualizer.selected_tele_columns:
            st.warning("No telemetry columns selected")
        else:
            st.warning("No telemetry data matches the current filters")

# Run the app when the script is executed directly
if __name__ == "__main__":
    run_streamlit_app()