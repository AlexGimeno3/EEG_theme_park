import datetime as dt
import pandas as pd

def get_datetime_from_excel(excel_path, id_col_name, id, time_column_name, date_column_name=None):
    """
    Function whose goal it is to return a relevant time value from an auxiliary Excel file.

    Inputs:
    - excel_path (Path object): path to the Excel sheet containing start time data
    - id_col_name (str): name of the ID column that we will find id (below) in
    - id (str): ID whose datetime we are looking for
    - time_column_name (str): name of the column containing the time (in the format HHMMSS) we are interested in
    - date_column_name (str or None): name of the column containing the date (in the format DDMMYYYY) we are interested in. If None, date is set to 1 Jan, 2000.
    
    Outputs:
    - start_datetime (dt.datetime object): the datetime object associated with the ID passed
    """
    
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Check if required columns exist
    if id_col_name not in df.columns:
        raise ValueError(f"ID column '{id_col_name}' not found in Excel file")
    if time_column_name not in df.columns:
        raise ValueError(f"Time column '{time_column_name}' not found in Excel file")
    if date_column_name is not None and date_column_name not in df.columns:
        raise ValueError(f"Date column '{date_column_name}' not found in Excel file")
    
    # Find the row with matching ID
    matching_rows = df[df[id_col_name] == id]
    
    if len(matching_rows) == 0:
        raise ValueError(f"ID '{id}' not found in column '{id_col_name}'")
    if len(matching_rows) > 1:
        raise ValueError(f"Multiple rows found with ID '{id}' in column '{id_col_name}'")
    
    # Extract time value
    time_value = matching_rows[time_column_name].iloc[0]
    
    # Convert to string and validate time format (HHMMSS - 6 digits)
    time_str = str(int(time_value)) if pd.notna(time_value) else None
    if time_str is None:
        raise ValueError(f"Time value is missing for ID '{id}'")
    
    # Pad with leading zeros if necessary (e.g., if time is stored as integer 123456)
    time_str = time_str.zfill(6)
    
    if len(time_str) != 6 or not time_str.isdigit():
        raise ValueError(f"Time value '{time_str}' for ID '{id}' is not in HHMMSS format (must be 6 digits)")
    
    # Validate hour, minute, second ranges
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    
    if hours > 23:
        raise ValueError(f"Invalid hour value {hours} in time '{time_str}'")
    if minutes > 59:
        raise ValueError(f"Invalid minute value {minutes} in time '{time_str}'")
    if seconds > 59:
        raise ValueError(f"Invalid second value {seconds} in time '{time_str}'")
    
    # Extract and validate date
    if date_column_name is not None:
        date_value = matching_rows[date_column_name].iloc[0]
        date_str = str(int(date_value)) if pd.notna(date_value) else None
        
        if date_str is None:
            raise ValueError(f"Date value is missing for ID '{id}'")
        
        # Pad with leading zeros if necessary
        date_str = date_str.zfill(8)
        
        if len(date_str) != 8 or not date_str.isdigit():
            raise ValueError(f"Date value '{date_str}' for ID '{id}' is not in DDMMYYYY format (must be 8 digits)")
        
        # Extract day, month, year
        day = int(date_str[0:2])
        month = int(date_str[2:4])
        year = int(date_str[4:8])
        
        # Validate ranges
        if day < 1 or day > 31:
            raise ValueError(f"Invalid day value {day} in date '{date_str}'")
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month value {month} in date '{date_str}'")
        if year < 1:
            raise ValueError(f"Invalid year value {year} in date '{date_str}'")
        
    else:
        # Default date: 1 Jan 2000
        day, month, year = 1, 1, 2000
    
    # Create datetime object
    try:
        start_datetime = dt.datetime(year, month, day, hours, minutes, seconds)
    except ValueError as e:
        raise ValueError(f"Invalid date/time combination: {e}")
    
    return start_datetime