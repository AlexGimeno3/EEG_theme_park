"""
This module contains code allowing the user to perform batch processing of EEG files. 
"""
from eeg_theme_park.modes.mode_manager import Mode, ModeManager
from eeg_theme_park.utils import gui_utilities, eeg_analyzers, loaders
from eeg_theme_park.modes import playground  
from eeg_theme_park.modes.batch_processing.batch_utils import verify_excel_path, select_flag_names, _get_file_names, get_flag_times
import os
from pathlib import Path
import pandas as pd
from eeg_theme_park.modes.batch_processing.pipeline_gui import build_pipeline, import_pipeline
import pickle
import tkinter as tk
from tkinter import ttk

class BatchProcessing(Mode):
    """
    Batch processing mode for importing data.
    """
    name = "batch_processing"

    def __init__(self, parent, mode_manager, **kwargs):
        """
        - keywords kwargs may contain that will be used in this code: "files_dir" (Path object with the )
        """
        super().__init__(parent, mode_manager)
        #Add state variables as needed here
        self.files_dir = kwargs.get("files_dir", None)
        if self.files_dir is not None:
            self.files_dir = Path(self.files_dir)
        self.events_excel_path = kwargs.get("excel_path", None)
        self.excel_path = self.events_excel_path
        self.event_names = kwargs.get("event_names", None)
        self.pipeline = None
        self.results_df = None
        self.errors_df = None
        self.analysis_log = ""
        self.gui_mode = kwargs.get("gui_mode", True)
        self._quality_control_done = False  # Track if we've done quality control

    def show(self):       
        # Call parent's show method
        super().show()

    def get_file_names(self):
        if not(self.event_names is None and self.excel_path is None):
            names_list, error_log_text = _get_file_names(self) #names_list is, as strings, all useable file names
        else:
            names_list = [f.name for f in self.files_dir.iterdir() if f.suffix in self.supported_extensions and f.is_file()]
            error_log_text = ""
        self.analysis_log += error_log_text
        self.names_list = [self.files_dir/Path(name) for name in names_list] #Now contains paths
    
    def initialize_ui(self):
        """
        Initializes UI elements for batch processing mode; called just once (the first time the mode is shown).
        """
        # Create main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create title
        title_label = ttk.Label(
            main_frame, 
            text="Batch Processing Mode", 
            font=('TkDefaultFont', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # 1. Choose Files section
        files_frame = ttk.LabelFrame(main_frame, text="1. Select Files", padding=15)
        files_frame.pack(fill=tk.X, pady=10)
        
        self.files_label = ttk.Label(files_frame, text="No directory selected")
        self.files_label.pack(pady=5)
        
        choose_files_btn = ttk.Button(
            files_frame, 
            text="Choose Files", 
            command=self._choose_files_command
        )
        choose_files_btn.pack(pady=5)
        
        # 2. Choose Events section
        events_frame = ttk.LabelFrame(main_frame, text="2. Select Events (Optional)", padding=15)
        events_frame.pack(fill=tk.X, pady=10)
        
        self.events_label = ttk.Label(events_frame, text="No events selected - will analyze entire signals")
        self.events_label.pack(pady=5)
        
        choose_events_btn = ttk.Button(
            events_frame,
            text="Choose Events",
            command=self._choose_events_command
        )
        choose_events_btn.pack(pady=5)
        
        # 3. Extract Features section
        extract_frame = ttk.LabelFrame(main_frame, text="3. Extract Features", padding=15)
        extract_frame.pack(fill=tk.X, pady=10)
        
        self.pipeline_label = ttk.Label(extract_frame, text="No pipeline loaded")
        self.pipeline_label.pack(pady=5)
        
        # Button frame for Extract Features buttons
        button_frame = ttk.Frame(extract_frame)
        button_frame.pack(pady=5)
        
        import_pipeline_btn = ttk.Button(
            button_frame,
            text="Import Pipeline",
            command=self._import_pipeline_command
        )
        import_pipeline_btn.pack(side=tk.LEFT, padx=5)
        
        self.extract_features_btn = ttk.Button(
            button_frame,
            text="Extract Features",
            command=self._extract_features_command,
            state='disabled'
        )
        self.extract_features_btn.pack(side=tk.LEFT, padx=5)

    def _choose_files_command(self):
        """Command handler for Choose Files button"""
        self.get_files_dir()
        if self.files_dir is not None:
            self.get_file_names()
            num_files = len(self.names_list) if hasattr(self, 'names_list') else 0
            self.files_label.config(text=f"Selected: {self.files_dir} ({num_files} files)")
            # Enable extract features button if files are selected
            self.extract_features_btn.config(state='normal')

    def _choose_events_command(self):
        """Command handler for Choose Events button"""
        self.excel_path = verify_excel_path(excel_path=None)
        if self.excel_path is not None:
            self.event_names = select_flag_names(self.excel_path)
            if self.event_names is not None:
                self.events_label.config(text=f"Events: {', '.join(self.event_names)}")
            else:
                self.events_label.config(text="No events selected - will analyze entire signals")
        else:
            self.events_label.config(text="No Excel file selected - will analyze entire signals")

    def _import_pipeline_command(self):
        """Command handler for Import Pipeline button"""
        imported_pipeline = import_pipeline()
        if imported_pipeline is not None:
            self.pipeline = imported_pipeline
            # Count operations in pipeline
            num_ops = len(self.pipeline.operations)
            self.pipeline_label.config(text=f"Pipeline loaded with {num_ops} operations")

    def _extract_features_command(self):
        """Command handler for Extract Features button"""
        if self.pipeline is None:
            self.pipeline = build_pipeline()
        
        if self.pipeline is not None:
            # Update label to show pipeline is being used
            if not hasattr(self, 'pipeline_label_updated') or not self.pipeline_label_updated:
                num_ops = len(self.pipeline.operations)
                self.pipeline_label.config(text=f"Using pipeline with {num_ops} operations")
                self.pipeline_label_updated = True
            
            self.run_analysis()
            
            # Show completion message
            gui_utilities.simple_dialogue(
                f"Analysis complete! Results saved to {self.files_dir}\n"
                f"Files processed: {len(self.names_list)}\n"
                f"Check the results Excel file for output."
            )

    def get_file_names(self):
        """Get list of valid file names from the selected directory"""
        if not(self.event_names is None and self.excel_path is None):
            names_list, error_log_text = _get_file_names(self)
        else:
            names_list = [f.name for f in self.files_dir.iterdir() if f.suffix in self.supported_extensions and f.is_file()]
            error_log_text = ""
        self.analysis_log += error_log_text
        self.names_list = [self.files_dir/Path(name) for name in names_list]
    
    def run_analysis(self, flag_times = None, save_pipeline=True):
        """
        Method that runs our analysis. Specifically, 
        - save_pipeline (bool): If True, we will save a pickled Pipeline object to the same directory where we save the Excel
        """
        #Step 1: Get our pipeline
        if self.pipeline is None:
            self.pipeline = build_pipeline()
        #Step 2: Initialize our results and error excels
        self.initialize_excel() #Initialize Excel based on pipeline's columns
        #Step 3: Load our files one-by-one (to save memory overhead) and runs our pipeline, saving as we go
        for path in self.names_list:
            eeg_signal = playground.file_commands.load_signal(file_path=path, file_name_bool = True)[0]
            id = eeg_signal.name
            #Add flag information
            if not flag_times is None:
                flag_times = get_flag_times(self.excel_path, id, self.event_names) #2-item arr of dt.time object representing the times between which the event occurred.
                flag_times = eeg_signal.get_real_time_window(flag_times)
                eeg_signal.analyze_time_limits = flag_times #Sets our analysis limits

            dicts_arr, error_message = self.pipeline.run_pipeline(eeg_signal) #Array, where each entry is a dictionary with the keys and results
            if dicts_arr is not None:
                # Transform the array of dicts into a single row
                row_data = {'ID': id}
                for result_dict in dicts_arr:
                    analyzer_name = result_dict['analyzer_name']
                    # Find matching column definitions
                    for col_def in self.column_defs:
                        if (col_def.get('analyzer_name') == analyzer_name and 
                            col_def.get('statistic') in result_dict):
                            stat_key = col_def['statistic']
                            col_name = col_def['full_name']
                            row_data[col_name] = result_dict[stat_key]
                
                # Append row to results DataFrame
                self.results_df = pd.concat([self.results_df, pd.DataFrame([row_data])], ignore_index=True)
            else:
                # Add to errors DataFrame
                error_row = {'ID': id, 'Error_name': error_message}
                self.errors_df = pd.concat([self.errors_df, pd.DataFrame([error_row])], ignore_index=True)
       
        #Step 3: Save results and errors DataFrames to Excel files
        base_results_path = self.files_dir / "eeg_theme_park_results.xlsx"
        base_errors_path = self.files_dir / "eeg_theme_park_errors.xlsx"
        # Function to get unique file path
        def get_unique_path(base_path):
            if not base_path.exists():
                return base_path
            
            stem = base_path.stem
            suffix = base_path.suffix
            parent = base_path.parent
            counter = 1
            
            while True:
                new_path = parent / f"{stem}_{counter}{suffix}"
                if not new_path.exists():
                    return new_path
                counter += 1
        # Get unique paths and save
        results_path = get_unique_path(base_results_path)
        errors_path = get_unique_path(base_errors_path)
        self.results_df.to_excel(results_path, index=False)
        self.errors_df.to_excel(errors_path, index=False)
        
        #Step 4: Optionally save pipeline
        if save_pipeline:
            pipeline_path = get_unique_path(self.files_dir / "pipeline.ppl")
            with open(pipeline_path, 'wb') as f:
                pickle.dump(self.pipeline, f)
        
        #Step 5: Save analysis log
        self.analysis_log += self.pipeline.pipeline_log
        log_path = get_unique_path(self.files_dir / "analysis_log.txt")
        with open(log_path, 'w') as f:
            f.write(self.analysis_log)
         
    def update_display(self):
        pass
    
    def get_files_dir(self):
        """
        This function will create a GUI window to allow the user to select the directory with files to analyze.

        Outputs:
        - None. However, once finished, will update self.files_dir.
        """ 
        chosen_files_dir = gui_utilities.choose_dir("Please choose the folder where you have the EEG files you would like to analyze.")
        if chosen_files_dir is None:
            return None
        all_files = [f for f in os.listdir(chosen_files_dir) if os.path.isfile(os.path.join(chosen_files_dir, f))]
        unsupported_files = [f for f in all_files if not any(f.endswith(ext) for ext in self.supported_extensions)]
        if unsupported_files:
            cont_bool = gui_utilities.simple_dialogue(
                f"Please note that the folder you select must contain only supported EEG files; it may not contain any other files. Currently, the formats {self.supported_extensions} are supported. Would you like to try again? (Yes to choose again, No to close this window).")
            if cont_bool:
                return self.get_files_dir()
        else:
            self.files_dir = chosen_files_dir
        return None

    def initialize_excel(self):
        """
        Code that initializes the results and error Excels for running the pipeline.
        
        Outputs:
        - results_df, errors_df (pandas df): dataframes with columns initialized for eventual export to excel
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be set before initializing Excel")
        
        # Get column definitions from pipeline
        column_defs = self.pipeline.get_expected_columns()
        
        # Extract just the full names with units for the DataFrame
        columns = [col_def['full_name'] for col_def in column_defs]
        
        # Store the column definitions for later use when filling in data
        self.column_defs = column_defs
        self.results_df = pd.DataFrame(columns=columns)
        
        # Build errors excel
        columns_error = ["ID", "Error_name"]
        self.errors_df = pd.DataFrame(columns=columns_error)
    
