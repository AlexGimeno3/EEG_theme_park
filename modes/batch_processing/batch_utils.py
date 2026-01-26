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
        
       # Extract ID column and the event columns (2 times + 2 optional dates)
        cols_to_extract = [id_col_str, my_mode.event_names[0], my_mode.event_names[1]]
        if my_mode.event_names[2] is not None:
            cols_to_extract.append(my_mode.event_names[2])
        if my_mode.event_names[3] is not None:
            cols_to_extract.append(my_mode.event_names[3])
        flag_df = df[cols_to_extract].copy()
        
        # Warn if dates are None
        if my_mode.event_names[2] is None or my_mode.event_names[3] is None:
            gui_utilities.simple_dialogue(
                "Warning: No date columns selected. This assumes all events occur on the same calendar day as the recording start. "
                "If your events span multiple days, please restart and select date columns."
            )
        
        # Check that time entries are in HHMMSS format and date entries are in DDMMYYYY format
        format_errors = []

        for idx, row in flag_df.iterrows():
            # Validate start time
            time_value = str(row[my_mode.event_names[0]])
            if not _is_valid_hhmmss(time_value):
                format_errors.append(f"{id_col_str}: {row[id_col_str]}, Start Time: {time_value}")
            
            # Validate end time
            time_value = str(row[my_mode.event_names[1]])
            if not _is_valid_hhmmss(time_value):
                format_errors.append(f"{id_col_str}: {row[id_col_str]}, End Time: {time_value}")
            
            # Validate start date if provided
            if my_mode.event_names[2] is not None:
                date_value = str(row[my_mode.event_names[2]])
                if not _is_valid_ddmmyyyy(date_value):
                    format_errors.append(f"{id_col_str}: {row[id_col_str]}, Start Date: {date_value}")
            
            # Validate end date if provided
            if my_mode.event_names[3] is not None:
                date_value = str(row[my_mode.event_names[3]])
                if not _is_valid_ddmmyyyy(date_value):
                    format_errors.append(f"{id_col_str}: {row[id_col_str]}, End Date: {date_value}")
        
        if format_errors:
            cont_bool = gui_utilities.yes_no(f"The following event times are not correctly formatted (expected HHMMSS) or are not present:\n" + "\n".join(format_errors)+"\n Knowing this, do you want to continue? Pressing yes will continue the analysis, but exclude these listed files.")
            if not cont_bool:
                return "Event times not correctly formatted", []
        
       # Check time/date consistency
        time_misformat = []
        for idx, row in flag_df.iterrows():
            time1 = int(str(row[my_mode.event_names[0]]))
            time2 = int(str(row[my_mode.event_names[1]]))
            
            # If dates are provided, check date order
            if my_mode.event_names[2] is not None and my_mode.event_names[3] is not None:
                date1 = int(str(row[my_mode.event_names[2]]))
                date2 = int(str(row[my_mode.event_names[3]]))
                
                # If end date is before start date, that's an error
                if date2 < date1:
                    time_misformat.append(str(row[id_col_str]))
                # If same date, check times
                elif date2 == date1 and time2 <= time1:
                    time_misformat.append(str(row[id_col_str]))
            else:
                # No dates provided - can only check times (same-day assumption)
                if time2 <= time1:
                    time_misformat.append(str(row[id_col_str]))
        
        if time_misformat:
            mismatch_cont_bool = gui_utilities.yes_no(
                f"The event times/dates in the IDs {time_misformat} appear invalid (end occurs before or at the same time as start). Would you like to proceed, knowing these {len(time_misformat)} files will be excluded?")
            if not mismatch_cont_bool:
                return "Time/date mismatch; operation ended by user", []
        
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
    Function that creates a GUI to allow the user to select event columns.

    Outputs:
    - flag_names (arr of str or None): 4-item array containing column names 
      [start_time_col, end_time_col, start_date_col, end_date_col]
      Date columns can be None if not selected.
    """
    
    # Load Excel file to get column names
    try:
        df = pd.read_excel(excel_path)
        column_names = df.columns.tolist()
    except Exception as e:
        gui_utilities.simple_dialogue(f"Error loading Excel file: {e}")
        return None
    
    # Create the result container
    result = {'flag_names': None}
    
    def on_submit():
        start_time = start_time_var.get()
        end_time = end_time_var.get()
        start_date = start_date_var.get() if start_date_var.get() != "None (same-day events)" else None
        end_date = end_date_var.get() if end_date_var.get() != "None (same-day events)" else None
        
        if start_time and end_time:
            result['flag_names'] = [start_time, end_time, start_date, end_date]
            window.destroy()
    
    def on_close():
        window.destroy()
    
    # Create window
    window = tk.Toplevel()
    window.title("Select Event Columns")
    window.protocol("WM_DELETE_WINDOW", on_close)
    
    # Instruction label
    tk.Label(window, text="Please select the columns containing event time and date data.\nDates are optional for same-day events (format: DDMMYYYY).", 
             wraplength=580).pack(pady=10)
    
    # Start time selection
    start_time_frame = tk.Frame(window)
    start_time_frame.pack(pady=5, padx=10, fill='x')
    tk.Label(start_time_frame, text="Start Time (HHMMSS):").pack(side='left')
    start_time_var = tk.StringVar()
    start_time_dropdown = ttk.Combobox(start_time_frame, textvariable=start_time_var, values=column_names, state='readonly')
    start_time_dropdown.pack(side='left', padx=5, fill='x', expand=True)
    
    # End time selection
    end_time_frame = tk.Frame(window)
    end_time_frame.pack(pady=5, padx=10, fill='x')
    tk.Label(end_time_frame, text="End Time (HHMMSS):").pack(side='left')
    end_time_var = tk.StringVar()
    end_time_dropdown = ttk.Combobox(end_time_frame, textvariable=end_time_var, values=column_names, state='readonly')
    end_time_dropdown.pack(side='left', padx=5, fill='x', expand=True)
    
    # Start date selection (with None option)
    start_date_frame = tk.Frame(window)
    start_date_frame.pack(pady=5, padx=10, fill='x')
    tk.Label(start_date_frame, text="Start Date (DDMMYYYY):").pack(side='left')
    start_date_var = tk.StringVar()
    date_options = ["None (same-day events)"] + column_names
    start_date_dropdown = ttk.Combobox(start_date_frame, textvariable=start_date_var, values=date_options, state='readonly')
    start_date_dropdown.set("None (same-day events)")
    start_date_dropdown.pack(side='left', padx=5, fill='x', expand=True)
    
    # End date selection (with None option)
    end_date_frame = tk.Frame(window)
    end_date_frame.pack(pady=5, padx=10, fill='x')
    tk.Label(end_date_frame, text="End Date (DDMMYYYY):").pack(side='left')
    end_date_var = tk.StringVar()
    end_date_dropdown = ttk.Combobox(end_date_frame, textvariable=end_date_var, values=date_options, state='readonly')
    end_date_dropdown.set("None (same-day events)")
    end_date_dropdown.pack(side='left', padx=5, fill='x', expand=True)
    
    # Submit button
    submit_button = tk.Button(window, text="Submit", command=on_submit)
    submit_button.pack(pady=10)
    
    # Wait for window to close
    window.wait_window()
    
    # If user closed without selecting times, ask if they want to continue
    if result['flag_names'] is None:
        cont_bool = gui_utilities.yes_no("You haven't selected event times. Processing will proceed with the entire EEG file. Is this OK?")
        if cont_bool:
            return None
        else:
            return select_flag_names(excel_path)
    
    return result['flag_names']


def _is_valid_ddmmyyyy(date_str):
    """Helper function to validate DDMMYYYY format."""
    if date_str is None or str(date_str).strip() in ['', 'nan', 'None']:
        return True  # None is acceptable
    date_str = str(date_str).strip()
    if len(date_str) != 8:
        return False
    if not date_str.isdigit():
        return False
    dd = int(date_str[0:2])
    mm = int(date_str[2:4])
    yyyy = int(date_str[4:8])
    return 1 <= dd <= 31 and 1 <= mm <= 12 and yyyy >= 1

def get_flag_times(flag_excel_path, id, flag_names):
    """
    Gets the flag start/end times and dates from the Excel
    
    Inputs:
    - flag_excel_path (Path object): path to the excel file
    - id (str): the key value for the eeg_signal file
    - flag_names (arr of str): 4-item array [start_time_col, end_time_col, start_date_col, end_date_col]
      Date columns can be None
    
    Outputs:
    - ret_arr (arr): 4-item array [dt.time, dt.time, dt.date or None, dt.date or None]
    """
    if not isinstance(flag_excel_path, Path):
        flag_excel_path = Path(flag_excel_path)
    
    try:
        # Load Excel file into pandas df
        excel_df = pd.read_excel(flag_excel_path)
        id_str = "ID"
        
        # Find the matching row
        matching_rows = excel_df[excel_df[id_str].astype(str) == str(id)]
        if matching_rows.empty:
            raise ValueError(f"No row found with ID '{id}' in the Excel file")
        
        row = matching_rows.iloc[0]
        
        # Extract time values
        start_time_value = row[flag_names[0]]
        end_time_value = row[flag_names[1]]
        
        # Convert times to dt.time objects
        def convert_to_time(value):
            """Helper function to convert HHMMSS format to datetime.time"""
            time_int = int(value)
            hours = time_int // 10000
            minutes = (time_int % 10000) // 100
            seconds = time_int % 100
            return dt.time(hours, minutes, seconds)
        
        start_time = convert_to_time(start_time_value)
        end_time = convert_to_time(end_time_value)
        
        # Extract and convert date values (if provided)
        start_date = None
        end_date = None
        
        if flag_names[2] is not None:
            start_date_value = row[flag_names[2]]
            if not pd.isna(start_date_value):
                date_int = int(start_date_value)
                day = date_int // 1000000
                month = (date_int % 1000000) // 10000
                year = date_int % 10000
                start_date = dt.date(year, month, day)
        
        if flag_names[3] is not None:
            end_date_value = row[flag_names[3]]
            if not pd.isna(end_date_value):
                date_int = int(end_date_value)
                day = date_int // 1000000
                month = (date_int % 1000000) // 10000
                year = date_int % 10000
                end_date = dt.date(year, month, day)
        
        ret_arr = [start_time, end_time, start_date, end_date]
        return ret_arr
        
    except Exception as e:
        raise ValueError(f"Error occurred ({e}), but should have been sandboxed. Please contact me at agimeno310@gmail.com") from e

    