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
        """
        self.name = signal_specs.get("name","unnamed")
        self.data = signal_specs["data"]
        self.original_data = copy.deepcopy(self.data) #Initial data in case the user ever wants to compare after filtering, noising, etc
        self.start_time = signal_specs["start_time"]
        self.time_collected = signal_specs.get("real_world_start_time", dt.datetime(2000, 1, 1, 0, 0, 0)) #Actual time that the recording began
        self.srate = signal_specs["srate"]
        self.end_time = self.start_time+len(self.data)/self.srate #In the edge case of one data point starting at t=0 sampled at 100 Hz, this means the signal start time is 0, the length is 1, and the end time is 0.01; this has been handled in the subsequent line of code 
        self.times = np.linspace(start=self.start_time, stop=self.end_time, num=len(self.data), endpoint=False) #Assuming a sampling rate of f, each delta-t is 1/f. self.end_time includes the time in the delta-t after the last data point collected at t. However, with linspace, this would cause incorrect time-marking creation, since the last data point would be assumed to be collected at self.end_time = t + delta-t when really it is just collected at t. Since the total recording length is always one delta-t above the total time that we are sampling, we use endpoint = False to eliminate this last delta-t
        self.time_collected = signal_specs.get("time_collected",dt.time(0, 0, 0)) #Time of the data point at time=0 as a dt.time object.
        self.display_time_lims = []
        self.analyze_time_lims = []
        self.time_series = [] #List of TimeSeries objects to allow for dynamic storage and generation
        self.flags = signal_specs.get("flags",{}) #Note that self.flags is in the format {"name":[list]}, where list contains one or two time values (IN MILLISECONDS). If list contains two time values, it must also contain a third boolean value (this is enforced with a setter); if shade is True, the space between the two displayed lines will be highlighted on all plots when displayed by the main GUI. 
        #Initialize log
        self.log = signal_specs.get("log","")
        current_datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log += f"[{current_datetime}] EEG file initialized using the specifications {signal_specs}\n\n"

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
            self._flags[normalized_name] = value
        elif len(value) == 2:
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
    
    def time_to_index(self, time, limit = None):
        """
        Function that takes in a time and finds the closest value (inclusive) to that value.

        Inputs:
        - time (float): time point in SECONDS that we would like to find the index of
        - limit (float): maximum difference we accept between the input time and the time value we were able to find (NOT inclusive). Default is 1/srate
        """
        #Initialize limit
        if limit is None:
            limit = 1/self.srate #Limit is within one sample if none supplied
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
        Function that returns the indices in self.data that correspond to a set of two flag times. Worth noting, flag times are "real world times"; therefore, to find the correct indices, we use self.time_collected.
        Inputs:
        - flag_times (arr of dt.time): two dt.time objects that we want to match
        """
        
        # Helper function to convert dt.time to seconds since midnight
        def time_to_seconds(time_obj):
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
        
        # Convert self.time_collected to seconds since midnight
        start_seconds = time_to_seconds(self.time_collected)
        
        # Create linearly spaced array starting at self.time_collected 
        # and advancing by increments of 1/self.srate seconds for as many points as exist in self.data
        time_increment = 1 / self.srate
        num_points = len(self.data)
        new_times_arr = start_seconds + np.arange(num_points) * time_increment
        
        # Convert flag_times to seconds since midnight
        flag_start_seconds = time_to_seconds(flag_times[0])
        flag_end_seconds = time_to_seconds(flag_times[1])
        
        # Find the points in new_times_arr that are closest to flag_times[0] and flag_times[1]
        start_idx = np.argmin(np.abs(new_times_arr - flag_start_seconds))
        end_idx = np.argmin(np.abs(new_times_arr - flag_end_seconds))
        
        start_point = new_times_arr[start_idx]
        end_point = new_times_arr[end_idx]
        
        # Check if either point is more than 2*1/srate secs away from the flag_time
        tolerance = 2 * time_increment
        
        if abs(start_point - flag_start_seconds) > tolerance or abs(end_point - flag_end_seconds) > tolerance:
            raise ValueError(f"It looks like this signal does not have data for those flags. Specifically, the flags you gave were {flag_times[0]} and {flag_times[1]}; however, this signal only has times ranging from {new_times_arr[0]} secs to {new_times_arr[-1]} secs.")
        
        # Return the from-zero time points of start_point and end_point
        return [self.times[start_idx], self.times[end_idx]]
        
        


    
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
    Each intstantiation of this class is both derived from and belongs to single EEGSiganl instance; these TimeSeries objects will be stored in each EEGSignal in the EEGSignal.time_series list.  
    """
    def __init__(self, **timeseries_specs):
        """
        Inputs
        - timeseries_specs: must include "name" (str, name of the extracted variable), "values" (float, the values of the extracted variable), "units" (str, the units for the extracted variable), "times" (foat, the times in ms that each value was recorded at); optional are "function" (stores the function object used to create the timeseries; good for reproducibility but heavier on storage)
        """
        #Quality checks and initialization
        necessary_keys = ["name","values","units","times","function"]
        if not all([key in timeseries_specs.keys() for key in necessary_keys]):
            raise ValueError(f"To build an EEGTimeSeries object, you must have at least the keys {necessary_keys}. However, the keys you passed were {timeseries_specs.keys()}")
        self.name = timeseries_specs["name"]
        if len(self.name)>0:
            name = self.name
        self.values = np.asarray(timeseries_specs["values"])
        self.units = timeseries_specs["units"].lower()
        self.times = np.asarray(timeseries_specs["times"])
        self.function = timeseries_specs["function"]

        if not len(self.times)==len(self.values):
            raise ValueError(f"Each value in the timeseries should have an associated time; however, there were {len(timeseries_specs["values"])} values and {len(timeseries_specs["times"])} times.")
    
    @property
    def data(self):
        return self.values
    
    @property
    def print_name(self):
        #Return self.name with just the first letter capitalized, and all underscores replaced with spaces
        if not self.name:
            return ""
        return self.name[0].upper() + self.name[1:].lower().replace('_', ' ')
