"""
Contains utilities we want to keep separated from the main Mode window; usually, these are ones we want to keep available for the user to use programatically.
"""
from pathlib import Path
from eeg_theme_park.utils import gui_utilities
import pandas as pd
import tkinter as tk
from tkinter import ttk
from eeg_theme_park.modes.playground import file_commands
import datetime as dt
import pandas as pd
from pathlib import Path
import datetime as dt

def run_analysis(eeg_names, my_mode):
    """
    This function will iterate over all signals in the eeg_names array (located at) and apply each analyzer in my_mode.analyzers to them. This will 

    NB: if a TimeSeries with a name corresponding to the analyzer name already exists in the file, this TimeSeries will be overwritten. Notifying the user of this will occur in the batch_processing.py file prior to the call of this function.
    
    Inputs:
    - eeg_names (arr of Path): Path objects containing the file name only of the eeg signals we will analyze 
    - my_mode (Mode subclass): the mode that we are running the analysis from
    """
    eeg_names = [my_mode.files_dir/name for name in eeg_names]
    for file in eeg_names:
        eeg_obj = file_commands.load_signal(file)
        
def _get_file_names(my_mode):
    """
    Function that uses the entries in excel_path as well as all_files_dir to determine which files to include in final analysis. Checks will include specifically ensuring all_files_dir files are listed in the Excel, Excel IDs are listed in all_files_dir, that events for each file exist, that events are formatted correctly, and that event 1's time is always before event 2's. Robust error logging is also incorporated throughout.

    Inputs:
    - my_mode (Mode subclass): Mode subclass from which we are calling check_excel

    Outputs:
    - error_message (str): the specific errors found
    - final_names (arr of str): an array containing the names of all the EEG files we will load (i.e., those that have been verified to have an Excel entry; checking/empty data handling will happen in a subsequent step.)
    """
    error_message = ""
    excel_path = my_mode.excel_path
    
    try:
        #Step 1: Load the excel file at excel_path
        df = pd.read_excel(excel_path)
        # Get our ID column name
        id_col_str = "ID"
        if "ID" not in df.columns:
            id_col_str = _select_id_column(df.columns.tolist())
            if id_col_str is None:
                return "No ID column selected", []
            # Rename the selected column to "ID" in the DataFrame
            df.rename(columns={id_col_str: "ID"}, inplace=True)
            # Write the modified DataFrame back to the Excel file
            df.to_excel(excel_path, index=False)
            # Update id_col_str to reflect the new column name
            id_col_str = "ID"
        excel_names = df[id_col_str].astype(str).tolist()
        
        #Step 2: Get file names and determine where there may be mismatches
        files_dir = Path(my_mode.files_dir)
        all_files = [f.stem for f in files_dir.iterdir() if f.is_file()]
        # Find files in excel but not in files_dir
        missing_from_files_dir = [name for name in excel_names if name not in all_files]
        # Find files in files_dir but not in excel
        missing_from_excel = [name for name in all_files if name not in excel_names]
        # Verify files in files_dir but missing from Excel
        if missing_from_excel:
            cont_bool = gui_utilities.yes_no(f"The following files are in the folder with EEG files but aren't listed in your Excel sheet: {missing_from_excel}. Would you like to proceed? Only the patients whose IDs are also listed in the Excel sheet will be processed.")
            if not cont_bool:
                return f"Missing Excel data for files {missing_from_excel}", []
        # Verify files in Excel but not in files_dir
        if missing_from_files_dir:
            cont_bool = gui_utilities.yes_no(f"The following IDs are in the Excel sheet but no corresponding files exist: {missing_from_files_dir}. Would you like to proceed? Only files that exist in both locations will be processed.")
            if not cont_bool:
                return f"Missing files for Excel entries {missing_from_files_dir}", []
        
        # Extract ID column and the two flag columns        
        flag_df = df[[id_col_str, my_mode.event_names[0], my_mode.event_names[1]]].copy()
        
        # Check that flag entries are in HHMMSS format
        format_errors = []

        for idx, row in flag_df.iterrows():
            for event_name in my_mode.event_names:
                time_value = str(row[event_name])
                if not _is_valid_hhmmss(time_value):
                    format_errors.append(f"{id_col_str}: {row[id_col_str]}, Event: {event_name}, Value: {time_value}")
        
        if format_errors:
            cont_bool = gui_utilities.yes_no(f"The following event times are not correctly formatted (expected HHMMSS) or are not present:\n" + "\n".join(format_errors)+"\n Knowing this, do you want to continue? Pressing yes will continue the analysis, but exclude these listed files.")
            if not cont_bool:
                return "Event times not correctly formatted", []
        
        # Check that event 2 time > event 1 time
        time_misformat = []
        for idx, row in flag_df.iterrows():
            time1 = int(str(row[my_mode.event_names[0]]))
            time2 = int(str(row[my_mode.event_names[1]]))
            if time2 <= time1:
                time_misformat.append(str(row[id_col_str]))
        
        if time_misformat:
            mismatch_cont_bool = gui_utilities.yes_no(f"EEG Theme Park cannot currently support recordings that have happened over multiple days; however, the second (later) event time in the IDs {time_misformat} seems to occur before the first one. This can happen when a recording is split across multiple days. Would you like to proceed, knowing these {len(time_misformat)} files will be excluded?")
            if not mismatch_cont_bool:
                return "Time mismatch; operation ended by user", []
        
        # Generate final ID array
        all_IDs = [f.stem for f in files_dir.iterdir() if f.is_file()]
        
        # Remove files missing from Excel
        for missing_id in missing_from_excel:
            if missing_id in all_IDs:
                all_IDs.remove(missing_id)
                error_message += f"Removed {missing_id}: not listed in Excel. "
        
        # Remove files with time formatting issues
        for error_id in format_errors:
            if error_id in all_IDs:
                all_IDs.remove(error_id)
                error_message += f"Removed {misformat_id}: event time mismatch. "
        
        for misformat_id in time_misformat:
            if misformat_id in all_IDs:
                all_IDs.remove(misformat_id)
                error_message += f"Removed {misformat_id}: event time mismatch. "
        
        # Refactor all entries in all_IDs to still be strings, but to get their file extensions back by searching in my_mode.files_dir
        all_IDs_with_extensions = []
        for file_id in all_IDs:
            matching_files = [f.name for f in files_dir.iterdir() if f.is_file() and f.stem == file_id]
            if matching_files:
                all_IDs_with_extensions.append(Path(matching_files[0]))
            else:
                raise ValueError("There was a file loading issue. Search the current error message to debug.")
        return all_IDs_with_extensions, error_message
        
    except Exception as e:
        gui_utilities.simple_dialogue(f"When trying to load the Excel file, there was an error: {e}. The program will work best if the file is input as a Path variable.")
        return str(e), []

def _select_id_column(column_names):
    """
    Helper function to create GUI for selecting ID column.

    Inputs:
    - column_names (list of str): list of all column names from the provided spreadsheet
    
    Outputs:
    - result["column"] (str): the name of the column containing the ID
    """
    result = {'column': None}
    def on_submit():
        result['column'] = column_var.get()
        window.destroy()
    def on_close():
        window.destroy()
    window = tk.Toplevel()
    window.title("Select ID Column")
    window.protocol("WM_DELETE_WINDOW", on_close)
    tk.Label(window, text="Please select the column containing the files' names (IDs). This will rename the column 'ID':").pack(pady=10)
    column_var = tk.StringVar()
    dropdown = ttk.Combobox(window, textvariable=column_var, values=column_names, state='readonly')
    dropdown.pack(pady=5, padx=10, fill='x')
    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack(pady=10)
    window.wait_window()
    return result['column']

def _is_valid_hhmmss(time_str):
    """Helper function to validate HHMMSS format."""
    time_str = str(time_str).strip()
    if len(time_str) != 6:
        return False
    if not time_str.isdigit():
        return False
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    ss = int(time_str[4:6])
    return 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59
    
def verify_excel_path(excel_path = None):
    """
    Function that both verifies a passed Excel path, and if it is not loadable, allows user to select a new Excel file via a GUI.

    Inputs:
    - excel_path (Path object): the Path of the selected Excel file
    
    Outputs:
    - Path object if successful, None if user chooses to proceed without Excel
    """
    if excel_path is None:
        excel_path = gui_utilities.get_file(file_types=[("Excel files", "*.xlsx")])
        if not excel_path:
            return None
    
    # Try loading the excel file
    try:
        df = pd.read_excel(excel_path)
        return Path(excel_path)
    except Exception as error_message:
        # If there is an error, ask user if they want to retry or continue without Excel
        reselect_bool = gui_utilities.yes_no(
            f"The Excel file you provided couldn't be loaded, showing the error: {error_message}. "
            "Would you like to select another file or just analyze the entirety of your EEG files? "
            "Yes will allow you to reselect a new Excel file, No will continue analysis with the entire EEG signals.")
        if reselect_bool:
            return verify_excel_path()
        else:
            return None

def select_flag_names(excel_path):
    """
    Function that creates a GUI to allow the user to select two flag names based the columns in the Excel files. NB: mode.event_names will be used to log which flags were saved.

    Outputs:
    - flag_names (arr of str): two-item array containing the names of the columns in excel_path that hold timing data for flag_1 and flag_2 for index 1 and index 2, respectively.
    """
    
    # Load Excel file to get column names
    try:
        df = pd.read_excel(excel_path)
        column_names = df.columns.tolist()
    except Exception as e:
        gui_utilities.simple_dialogue(f"Error loading Excel file: {e}")
        return None
    
    # Create the result container
    result = {'flag_names': None, 'should_continue': False}
    
    def on_submit():
        event1 = event1_var.get()
        event2 = event2_var.get()
        
        if event1 and event2:
            result['flag_names'] = [event1, event2]
            window.destroy()
    
    def on_close():
        window.destroy()
    
    # Create window
    window = tk.Toplevel()
    window.title("Select Event Columns")
    window.protocol("WM_DELETE_WINDOW", on_close)
    
    # Instruction label
    tk.Label(window, text="Please select the columns in the Excel file that contain the time event data.").pack(pady=10)
    
    # Event 1 selection
    event1_frame = tk.Frame(window)
    event1_frame.pack(pady=5, padx=10, fill='x')
    tk.Label(event1_frame, text="Event 1:").pack(side='left')
    event1_var = tk.StringVar()
    event1_dropdown = ttk.Combobox(event1_frame, textvariable=event1_var, values=column_names, state='readonly')
    event1_dropdown.pack(side='left', padx=5, fill='x', expand=True)
    
    # Event 2 selection
    event2_frame = tk.Frame(window)
    event2_frame.pack(pady=5, padx=10, fill='x')
    tk.Label(event2_frame, text="Event 2:").pack(side='left')
    event2_var = tk.StringVar()
    event2_dropdown = ttk.Combobox(event2_frame, textvariable=event2_var, values=column_names, state='readonly')
    event2_dropdown.pack(side='left', padx=5, fill='x', expand=True)
    
    # Submit button
    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack(pady=10)
    
    # Wait for window to close
    window.wait_window()
    
    # If user closed without selecting, ask if they want to continue
    if result['flag_names'] is None:
        cont_bool = gui_utilities.yes_no("You haven't selected a flag. Processing will proceed with the entire EEG file. Is this OK?")
        if cont_bool:
            return None
        else:
            return select_flag_names(excel_path)
    
    return result['flag_names']

def get_flag_times(flag_excel_path, id, flag_names):
    """
    Gets the flag start and end times (as a datetime.time object) from the Excel
    Inputs:
    - flag_excel_path (Path object): path to the excel file holding the flag names and values
    - id (str): the key value for the eeg_signal file we are getting the flag data from
    - flag_names (arr of str): 2-item array containing the column names of the flags
    Outputs:
    - ret_arr (arr of dt.time): 2-item array of dt.time objects that represent the start and end times for the flag
    """
    if not isinstance(flag_excel_path, Path):
        flag_excel_path = Path(flag_excel_path)
    
    try:
        # Load Excel file into pandas df
        excel_df = pd.read_excel(flag_excel_path)
        # The ID column should be named "ID"
        id_str = "ID"
        # Find the entry in excel_df where the column id_str equals id
        # Convert both to strings for comparison to handle different data types
        matching_rows = excel_df[excel_df[id_str].astype(str) == str(id)]
        if matching_rows.empty:
            raise ValueError(f"No row found with ID '{id}' in the Excel file")
        # Get the first matching row
        row = matching_rows.iloc[0]
        # Extract the values from the two flag columns
        flag_start_value = row[flag_names[0]]
        flag_end_value = row[flag_names[1]]
        # Convert each entry into a dt.time object
        def convert_to_time(value):
            """Helper function to convert HHMMSS format to datetime.time"""
            # Convert to int if it's a float or string
            time_int = int(value)
            # Extract hours, minutes, and seconds from HHMMSS format
            hours = time_int // 10000
            minutes = (time_int % 10000) // 100
            seconds = time_int % 100
            # Create and return the time object
            return dt.time(hours, minutes, seconds)
        # Convert both flag values to time objects
        start_time = convert_to_time(flag_start_value)
        end_time = convert_to_time(flag_end_value)
        # Store in return array
        ret_arr = [start_time, end_time]
        return ret_arr
        
    except Exception as e:
        # If any errors occur, raise the specified error message
        raise ValueError(f"Error occurred ({e}), but should have been sandboxed. Please contact me at agimeno310@gmail.com") from e

    