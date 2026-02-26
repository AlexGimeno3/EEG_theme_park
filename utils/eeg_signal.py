"""
Here, we have a class representing an EEG signal and another representing an EEG time series.
"""
import numpy as np
import copy
import datetime as dt
import inspect
from eeg_theme_park.utils.gui_utilities import simple_dialogue

class EEGSignal:
    def __init__(self, **signal_specs):
        """
        Initialization method to build the object.

        Inputs:
        - signal_specs (dict): a dictionary containing the following keys: "srate", "data", "start_time", "flags", "log"
          Optional: "all_channel_data" (dict {ch_name: np.array}), "all_channel_labels" (list of str)
        """
        self.name = signal_specs.get("name","unnamed")
        self.channel = signal_specs.get("channel", "unspecified")
        self.sourcer = signal_specs.get("sourcer","unspecified")
        
        # Multi-channel support
        if "all_channel_data" in signal_specs:
            self.all_channel_data = {k: np.asarray(v) for k, v in signal_specs["all_channel_data"].items()}
            self.all_channel_labels = signal_specs.get("all_channel_labels", list(self.all_channel_data.keys()))
        else:
            # Backward-compatible: build from single-channel data
            single_data = np.asarray(signal_specs["data"])
            ch_name = signal_specs.get("channel", "unspecified")
            self.all_channel_data = {ch_name: single_data}
            self.all_channel_labels = [ch_name]
        
        self.current_channel = self.channel  # The user-selected primary channel
        
        self.start_time = signal_specs["start_time"]
        _dt_collected = signal_specs.get("datetime_collected", dt.datetime(2000, 1, 1, 0, 0, 0))
        self.datetime_collected = _dt_collected.replace(tzinfo=None) if _dt_collected.tzinfo is not None else _dt_collected
        self.srate = signal_specs["srate"]
        self.end_time = self.start_time+len(self.data)/self.srate #In the edge case of one data point starting at t=0 sampled at 100 Hz, this means the signal start time is 0, the length is 1, and the end time is 0.01; this has been handled in the subsequent line of code 
        self.times = np.linspace(start=self.start_time, stop=self.end_time, num=len(self.data), endpoint=False) #Assuming a sampling rate of f, each delta-t is 1/f. self.end_time includes the time in the delta-t after the last data point collected at t. However, with linspace, this would cause incorrect time-marking creation, since the last data point would be assumed to be collected at self.end_time = t + delta-t when really it is just collected at t. Since the total recording length is always one delta-t above the total time that we are sampling, we use endpoint = False to eliminate this last delta-t
        self.display_time_lims = []
        self.analyze_time_lims = []
        self.time_series = [] #List of TimeSeries objects to allow for dynamic storage and generation
        self.flags = signal_specs.get("flags",{}) #Note that self.flags is in the format {"name":[list]}, where list contains one or two time values (IN MILLISECONDS). If list contains two time values, it must also contain a third boolean value (this is enforced with a setter); if shade is True, the space between the two displayed lines will be highlighted on all plots when displayed by the main GUI. 
        self._flag_visibility = {} #Dictionary mapping eeach flag name to a boolean to determine whether or not to show it in playground mode. If True, shown in playground mode; if False, not shown when rendering.
        #Initialize log
        self.log = signal_specs.get("log","")
        current_datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log += f"[{current_datetime}] EEG file initialized using the specifications {signal_specs}\n\n"

    @property
    def data(self):
        return self.all_channel_data[self.current_channel]

    @data.setter
    def data(self, value):
        self.all_channel_data[self.current_channel] = np.asarray(value)

    def switch_channel(self, new_channel):
        """
        Switch the primary channel. Updates current_channel and propagates
        primary_channel to all multi-channel TimeSeries objects.
        
        Inputs:
        - new_channel (str): channel name to switch to (must be in all_channel_labels)
        """
        if new_channel not in self.all_channel_labels:
            raise ValueError(f"Channel '{new_channel}' not found. Available channels: {self.all_channel_labels}")
        self.current_channel = new_channel
        self.channel = new_channel
        # Update end_time and times based on new channel's data length
        self.end_time = self.start_time + len(self.data) / self.srate
        self.times = np.linspace(start=self.start_time, stop=self.end_time, num=len(self.data), endpoint=False)
        # Propagate to all TimeSeries
        for ts in self.time_series:
            if ts.channel_data is not None:
                if new_channel in ts.channel_data:
                    ts.primary_channel = new_channel
                # If the channel isn't in this TimeSeries's channel_data,
                # leave primary_channel unchanged (it will still return
                # data for whatever channel it was last set to)
    
    def __setattr__(self, name, value):
        #Rewritten so changes to data are automatically logged; this allows users to add functions to eeg_functions.py without needing to add a .log() step in every one.
        if name == "data" and hasattr(self, "data"):
            stack = inspect.stack() #Get call stack
            altering_frame = stack[1] #stack[0] is __setattr__()
            altering_fxn = altering_frame.function
            altering_filename = altering_frame.filename
            altering_lineno = altering_frame.lineno
            #Generate and add log text
            current_datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            add_text = f"Data modified by the function {altering_fxn} at {altering_filename} on line {altering_lineno}."
            log_text = f"[{current_datetime}] {add_text}\n\n"
            object.__setattr__(self, "log", self.log + log_text) #Must be set this way to avoid an infinite loop of calling __setattr__
        
        object.__setattr__(self, name, value)

    def log_text(self, text: str):
        """
        Add text to the signal's change log.

        Inputs:
        - text (str): text to add
        """
        current_datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log += f"[{current_datetime}] {text}\n\n"

    def toggle_shade(self, flag_name):
        """
        Toggles the shade boolean of a given flag title.

        Inputes:
        - flag_name (str): key in the self.flags dictionary
        """
        # Normalize the flag name to match the setter's capitalization
        normalized_name = flag_name[0].upper() + flag_name[1:] if flag_name else flag_name
        if normalized_name not in self.flags.keys():
            raise KeyError(f"The flag you requested ({flag_name}) does not exist amongst the flags attached to the signal {self.name} ({self.flags.keys().tolist()})")
        else:
            self._flags[normalized_name][-1] = not self._flags[normalized_name][-1]

    def _get_signal_time(self, time_value):
        """
        Convert a datetime/time object to signal-relative time in seconds. Also accepts numeric values (returned unchanged).
        
        Inputs:
        - time_value: float/int (signal-relative seconds), dt.time, or dt.datetime
        
        Outputs:
        - float: signal-relative time in seconds from signal start
        
        Examples:
        - Input: 125.5 → Output: 125.5 (already signal-relative)
        - Input: dt.time(14, 32, 15) → Output: 135.0 (if recording started at 14:30:00)
        - Input: dt.datetime(2024, 1, 15, 14, 32, 15) → Output: 135.0 (assuming signal recorded on 15 Jan 2024)
        """
        # If already a number, assume it's signal-relative time
        if isinstance(time_value, (int, float, np.integer, np.floating)):
            return float(time_value)
        
        # Convert dt.time to dt.datetime by assuming it occurred on recording date
        if isinstance(time_value, dt.time):
            time_value = dt.datetime.combine(self.datetime_collected.date(), time_value)
        
        # Convert dt.datetime to signal-relative time
        if isinstance(time_value, dt.datetime):
            standardized_time = time_value.timestamp()
            standardized_start_time = self.datetime_collected.timestamp()
            
            # Calculate offset from recording start
            offset_seconds = standardized_time - standardized_start_time
            
            return offset_seconds - self.start_time #self.start_time represents the first time value in the signal relative to when to recording was started. If it equals 5, therefore, the signal begins 5 seconds after the recording was first taken. Therefore, incorporating this substraction ensures that our offset (from beginning of the recording) is corrected in those cases where some trimming has occurred. As a concrete example, imagine a signal recorded at 14:00:00, but cropped to 14:00:10 and with an event at 14:00:15. offset_seconds would be 15 (since the event happened 15 secs after the recording started); however, the event is only 5 seconds into the actual signal (which started at 14:00:10, and therefore has a start_time of 10 secs). 15-10 = 5 seconds, accurately reflecting the number of seconds from the signal start where we see our event. 
        
        raise TypeError(
            f"Flag time must be numeric (float/int) or datetime object (dt.time/dt.datetime), got {type(time_value).__name__}: {time_value}")
    
    def add_flag(self, flag_name, value, shade=False):
        """
        Adds flag to the eeg_signal object.

        Inputs:
        - flag_name (str): name of the flag we will add (this becomes the key in the dictionary)
        - value (list): list to assign as the key's value (each value is a time point; one value indicates just one time point, while two values indicates a time range)
        - shade (bool): if True and there are two other values in self.flags[flag_name], we will see shading over the time period; if false, no shading will be applied
        """
        normalized_name = flag_name[0].upper() + flag_name[1:] if flag_name else flag_name
        # If normalized_name is already in self._flags.keys(), add an integer to the end of normalized_name, ascending (e.g., _1, _2, ...) until there is no longer any overlap
        if not hasattr(self, '_flags'):
            self._flags = {}
        
        original_name = normalized_name
        counter = 1
        while normalized_name in self._flags:
            normalized_name = f"{original_name}_{counter}"
            counter += 1
        if not isinstance(value,list):
            value = value.tolist()
        value = value.copy()

        if len(value) == 1:
            self._flags[normalized_name] = [self._get_signal_time(entry) for entry in value]
        elif len(value) == 2:
            value = [self._get_signal_time(entry) for entry in value]
            value.append(shade)
            self._flags[normalized_name] = value
        else:
            raise ValueError(f"The time(s) you've specified for your flag is {value}; however, you may only have one time (for a fixed event) or two times (for an event lasting a certain duration).")
    
    def remove_flag(self, flag_name):
        """
        Removes flag from the eeg_signal object.

        Inputs:
        - flag_name (str): name of the flag to remove

        Outputs:
        None.
        """
        normalized_name = flag_name[0].upper() + flag_name[1:] if flag_name else flag_name
        if normalized_name not in self._flags:
            raise KeyError(f"The flag you requested ({flag_name}) does not exist amongst the flags in the signal ({self.name}). Available flags: {list(self._flags.keys())}.")
        del self._flags[normalized_name]
        self._flag_visibility.pop(normalized_name,None)
    
    def time_to_index(self, time, limit = None):
        """
        Function that takes in a time and finds the closest value (inclusive) to that value.

        Inputs:
        - time (float): time point in SECONDS that we would like to find the index of
        - limit (float): maximum difference we accept between the input time and the time value we were able to find (NOT inclusive). Default is 2/srate
        """
        #Initialize limit
        if limit is None:
            limit = 2/self.srate #Limit is within two samples if none supplied
        # Find value val in self.times closest to time
        idx = np.argmin(np.abs(self.times - time))
        val = self.times[idx]
        # Check if the closest value is within the limit
        if abs(val - time) > limit:
            simple_dialogue(f"You asked for the index at time point {time}; however, the closest time point to that in the signal is {val}, but the limit imposed is {limit} secs.")
            return None
        else:
            return idx
    
    def get_real_time_window(self, flag_times):
        """
        Function that converts real-world datetime objects to signal-relative times, finding the closest actual sample point for each.
        
        Inputs:
        - flag_times (list of float, dt.time, or dt.datetime): list of flag times
        
        Outputs:
        - converted_times (2-item list of floats): signal-relative times in seconds (snapped to actual samples)
        """
        # Convert to signal-relative times
        converted_times = []
        tolerance = 2 / self.srate
        
        for flag_time in flag_times:
            # Use the helper to get signal-relative time
            signal_time = self._get_signal_time(flag_time)
            
            # Find closest actual sample point
            idx = np.argmin(np.abs(self.times - signal_time))
            closest_time = self.times[idx]
            
            # Validate tolerance
            if abs(closest_time - signal_time) > tolerance:
                # Format flag_time for error message based on its type
                if isinstance(flag_time, dt.datetime):
                    flag_str = flag_time.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(flag_time, dt.time):
                    flag_dt = dt.datetime.combine(self.datetime_collected.date(), flag_time)
                    flag_str = flag_dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # It's a numeric value (float/int)
                    flag_str = f"{flag_time:.3f}s (signal-relative)"
                
                raise ValueError(f"Flag time {flag_str} (converts to signal time {signal_time:.3f}s) is not close to any sample point. Closest sample is at {closest_time:.3f}s (difference: {abs(closest_time - signal_time):.4f}s, tolerance: {tolerance:.4f}s). The signal's begins at {self.datetime_collected}.")
            
            converted_times.append(closest_time)
        
        return converted_times
        
        


    
    @property
    def analyze_time_limits(self): #Added for redundancy
        return self.analyze_time_lims
    @analyze_time_limits.setter
    def analyze_time_limits(self, value):
        self.analyze_time_lims = value
    
    @property
    def flags(self):
        return self._flags
    @flags.setter
    def flags(self,right_of_equals):
        final_flags = {}
        for k,v in right_of_equals.items():
            k = k[0].upper() + k[1:]
            if len(v) == 2: #If there are only 2 items, this implies limits of an event without shading information
                val = [v[0],v[1],False] #Automatically sets last value to False, which handles the shading
                final_flags[k]=val
            else: #1 entry is just a single event, 3 implies we have start and end limits with a shading boolean
                final_flags[k] = v
        self._flags = final_flags

class TimeSeries:
    """
    Each instantiation of this class is both derived from and belongs to a single EEGSignal instance; these TimeSeries objects will be stored in each EEGSignal in the EEGSignal.time_series list.

    Multi-channel support:
    - If channel_data is None, this is a legacy/single-value TimeSeries (e.g., from a multi-channel analyzer like coherence). values and times are simple arrays.
    - If channel_data is populated, it is a dict of the form {ch_name: {"values": np.array, "times": np.array}} containing per-channel results. The `values` and `times` properties route to the entry corresponding to `primary_channel`.
    """
    def __init__(self, **timeseries_specs):
        """
        Inputs
        - timeseries_specs: must include "name", "values", "units", "times", "function".
          Optional: "channel_data" (dict), "primary_channel" (str).
        """
        necessary_keys = ["name", "values", "units", "times", "srate", "function"]
        if not all(key in timeseries_specs.keys() for key in necessary_keys):
            raise ValueError(f"To build a TimeSeries object, you must have at least the keys {necessary_keys}. However, the keys you passed were {list(timeseries_specs.keys())}")
        
        self.name = timeseries_specs["name"]
        self._values = np.asarray(timeseries_specs["values"])
        self.units = timeseries_specs["units"].lower()
        self._times = np.asarray(timeseries_specs["times"])
        self.function = timeseries_specs["function"]
        self.srate = timeseries_specs["srate"]
        
        # Multi-channel support
        self.channel_data = timeseries_specs.get("channel_data", None)  # {ch_name: {"values": np.array, "times": np.array}} or None
        self.primary_channel = timeseries_specs.get("primary_channel", None)

        if self.channel_data is None:
            if len(self._times) != len(self._values):
                raise ValueError(f"Each value in the timeseries should have an associated time; however, there were {len(self._values)} values and {len(self._times)} times.")

    @property
    def values(self):
        if self.channel_data is not None and self.primary_channel is not None:
            return self.channel_data[self.primary_channel]["values"]
        return self._values

    @values.setter
    def values(self, new_values):
        if self.channel_data is not None and self.primary_channel is not None:
            self.channel_data[self.primary_channel]["values"] = np.asarray(new_values)
        else:
            self._values = np.asarray(new_values)

    @property
    def times(self):
        if self.channel_data is not None and self.primary_channel is not None:
            return self.channel_data[self.primary_channel]["times"]
        return self._times

    @times.setter
    def times(self, new_times):
        if self.channel_data is not None and self.primary_channel is not None:
            self.channel_data[self.primary_channel]["times"] = np.asarray(new_times)
        else:
            self._times = np.asarray(new_times)

    def get_channel_values(self, channel_name):
        """Retrieve the values array for a specific channel."""
        if self.channel_data is None:
            raise ValueError("No multi-channel data stored in this TimeSeries.")
        if channel_name not in self.channel_data:
            raise KeyError(f"Channel '{channel_name}' not found. Available: {list(self.channel_data.keys())}")
        return self.channel_data[channel_name]["values"]

    def get_channel_times(self, channel_name):
        """Retrieve the times array for a specific channel."""
        if self.channel_data is None:
            raise ValueError("No multi-channel data stored in this TimeSeries.")
        if channel_name not in self.channel_data:
            raise KeyError(f"Channel '{channel_name}' not found. Available: {list(self.channel_data.keys())}")
        return self.channel_data[channel_name]["times"]
    
    @property
    def data(self):
        return self.values
    
    @property
    def print_name(self):
        #Return self.name with just the first letter capitalized, and all underscores replaced with spaces
        if not self.name:
            return ""
        return self.name[0].upper() + self.name[1:].lower().replace('_', ' ')
