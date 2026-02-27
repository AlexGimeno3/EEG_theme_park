"""
This contains the different functions we may want to apply to our signals. It should be built to handle all kinds of scenarios; filtering, adding noise, adding features. Each function should take in the eeg_signal_obj of interest, as well as the fxn_lims (i.e., the start and end time of the function we would like to apply), flags (i.e., do we want to add flags to where we did this so they are displayed in the viewer?) and **kwargs  
"""

from abc import ABC, abstractmethod
import numpy as np
import inspect
from eeg_theme_park.utils.gui_utilities import simple_dialogue
from eeg_theme_park.utils.NEURAL_py_fork.preprocessing_EEG import art_per_channel
import tkinter as tk
from tkinter import ttk
import copy
from scipy.signal import butter, filtfilt, sosfiltfilt
from pathlib import Path
import matplotlib.pyplot as plt

class EEGFunction(ABC):
    """
    Abstract base class for EEG signal processing functions.
    """
    @property
    @abstractmethod
    def name(self):
        """
        Each subclass must define a name property.
        """
        pass

    @property
    @abstractmethod
    def params_units_dict(self):
        """
        Each subclass must define a params_units_dict property.
        """
        pass

    def __init__(self, **kwargs):
        """
        When initializing a new function.

        Input:
        - name (str): name of the function we are adding
        """
        self.args_dict = kwargs
    
    def __init_subclass__(cls, **kwargs):
        #Ensure that, for every EEGFunction subclass that is defined below, the subclass is immediately registered in AllFunctions and therefore accessible via AllFunctions._functions
        super().__init_subclass__(**kwargs)
        AllFunctions.add_to_functions(cls)
    
    def apply(self, eeg_object, time_range=None, flags_bool=True, min_clean_length=0):
        """
        Template method to apply the given function to a signal.
        Loops over ALL channels in eeg_object.all_channel_data.

        Inputs:
        - eeg_object (EEGSignal object): EEGSignal object
        - time_range (list): time range in seconds. If None and analyze_time_lims is set, uses analyze_time_lims. Otherwise uses entire signal.
        - flags_bool (bool): if True, will add flags
        - min_clean_length (float): minimum length (in seconds) of clean segments to process

        Output:
        - eeg_object (EEGSignal object): EEGSignal object after change has been applied
        """
        # Determine time range to process
        if time_range is None:
            if len(eeg_object.analyze_time_lims) > 0:
                time_range = [eeg_object.analyze_time_lims[0], eeg_object.analyze_time_lims[1]]
            else:
                time_range = [eeg_object.times[0], eeg_object.times[-1]]

        try:
            start_idx = eeg_object.time_to_index(time_range[0])
            end_idx = eeg_object.time_to_index(time_range[1])
        except Exception as e:
            raise type(e)(
                f"{str(e)}; the time range you asked to apply the function to was "
                f"{time_range[0]}-{time_range[1]} secs, but the signal only goes from "
                f"{eeg_object.start_time} to {eeg_object.end_time} secs."
            ) from e

        original_channel = eeg_object.current_channel  # Save to restore

        for ch_name in eeg_object.all_channel_labels:
            eeg_object.current_channel = ch_name  # Routes eeg_object.data to this channel

            if min_clean_length > 0:
                from eeg_theme_park.utils.pipeline import find_clean_segments
                data_to_process = eeg_object.data[start_idx:end_idx + 1]
                clean_segments = find_clean_segments(data_to_process, eeg_object.srate, min_clean_length)

                processed_data = np.full_like(data_to_process, np.nan)
                for seg_start, seg_end in clean_segments:
                    segment_data = data_to_process[seg_start:seg_end]
                    processed_segment = self._apply_function(segment_data, eeg_object)
                    processed_data[seg_start:seg_end] = processed_segment

                eeg_object.data[start_idx:end_idx + 1] = processed_data
            else:
                data_to_process = eeg_object.data[start_idx:end_idx + 1]
                edited_data = self._apply_function(data_to_process, eeg_object)
                eeg_object.data[start_idx:end_idx + 1] = edited_data

        eeg_object.current_channel = original_channel  # Restore

        # Add flag once (not per channel)
        if flags_bool:
            eeg_object.add_flag(self.name, copy.deepcopy(time_range))
        eeg_object.has_unsaved_changes = True

        return eeg_object

    
    @abstractmethod
    def _apply_function(self, original_signal, eeg_object, **kwargs):
        """
        Abstract method that must be used when creating a new function. Sublclasses should contain all the code that modifies the signal.

        Inputs:
        - original_signal (list of int): the original signal (in uV) that we would like to modify
        - eeg_object (EEGSignal object): the EEG signal object from which we may want to pull data (e.g., sampling rate)

        Output:
        - modified_signal (list of int): the modified signal (in uV) after the function has been applied
        """
        pass
    
    @classmethod
    def get_params(cls, eeg_object, parent=None):
        """
        Creates a dialogue to collect parameter values from the user based on the __init__() method's signature (as defined in the subclass).

        Inputs:
        - eeg_object (eeg_object): eeg_object we will use to get srate

        Outputs:
        - ret_var (dict): a dictionary of values in the format {"param_name":value}
        """
        sig = inspect.signature(cls.__init__) #Accesses the subclass's __init__ signature
        params_to_collect = {name: param for name, param in sig.parameters.items() if not name in ('self','kwargs') and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)} #Gets our parameters, filtering out special ones
        if len(params_to_collect) == 0:
            return {}
        ret_var = {}

        def check_fields(*args):
            """
            Ensures required fields (i.e., all params) are filled 
            """
        # Enable submit if all required fields (no defaults) have values
        all_required_filled = True
        for param_name, param in params_to_collect.items():
            if param.default is inspect.Parameter.empty: #I.e., if there is no default value for a given parameter
                # This is a required field
                if string_vars[param_name].get().strip() == "":
                    all_required_filled = False
                    break
        

        def submit():
            nonlocal ret_var
            # Collect values from entry boxes
            for param_name, str_var in string_vars.items():
                value_str = str_var.get().strip()
                if value_str == "":
                    # Check if this parameter has a default value
                    param = params_to_collect[param_name]
                    if param.default is inspect.Parameter.empty:
                        simple_dialogue(f"Parameter '{param_name}' is required but was not provided.")
                        return
                    else:
                        # Use default value
                        ret_var[param_name] = param.default
                else:
                    # Try to convert to appropriate type
                    param = params_to_collect[param_name]
                    annotation = param.annotation
                    
                    try:
                        # Check if user explicitly entered None
                        if value_str.lower() in ('none', 'null', ''):
                            ret_var[param_name] = None
                        elif annotation == int or (annotation == inspect.Parameter.empty and 
                                                param.default is not inspect.Parameter.empty and 
                                                isinstance(param.default, int)):
                            ret_var[param_name] = int(value_str)
                        elif annotation == float or (annotation == inspect.Parameter.empty and 
                                                    param.default is not inspect.Parameter.empty and 
                                                    isinstance(param.default, float)):
                            ret_var[param_name] = float(value_str)
                        elif annotation == bool:
                            ret_var[param_name] = value_str.lower() in ('true', '1', 'yes')
                        else:
                            # Try to convert to float for numeric strings
                            try:
                                ret_var[param_name] = float(value_str)
                            except ValueError:
                                # If float conversion fails, keep as string
                                ret_var[param_name] = value_str
                    except ValueError:
                        simple_dialogue(f"Could not convert '{value_str}' to appropriate type for parameter '{param_name}'.")
                        return
            
            dialogue.destroy() 

        #Build main box
        dialogue = tk.Toplevel(parent)
        dialogue.title(f"Parameters for {cls.name}")
        dialogue.geometry("600x400")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()
        main_label = ttk.Label(dialogue, text=f"Please enter parameters for '{cls.name}'", wraplength=580)
        main_label.grid(column=0, row=0, columnspan=2, padx=10, pady=10)  

        string_vars = {}
        row = 1
        units_dict = cls.params_units_dict

        #Create entry boxes
        for param_name, param in params_to_collect.items():
            # Create label
            label_text = param_name.replace('_', ' ').title()
            if param_name in units_dict.keys():
                label_text+=f" ({units_dict[param_name]})"
            label = ttk.Label(dialogue, text=f"{label_text}:")
            label.grid(column=0, row=row, sticky="w", padx=10, pady=5)
            
            # Create entry box
            str_var = tk.StringVar(dialogue)
            if param.default is not inspect.Parameter.empty:
                str_var.set(str(param.default))
            if param_name == 'srate':
                str_var.set(str(eeg_object.srate))
            str_var.trace_add('write', check_fields)
            entry = ttk.Entry(dialogue, textvariable=str_var, width=30)
            entry.grid(column=1, row=row, padx=10, pady=5)
            string_vars[param_name] = str_var
            row += 1

        # Submit button
        submit_button = ttk.Button(dialogue, text="Submit", command=submit, state='disabled')
        submit_button.grid(column=0, columnspan=2, row=row, pady=20)

        submit_button.config(state='normal' if all_required_filled else 'disabled')
        check_fields()
        dialogue.wait_window()
        return ret_var if ret_var else None

    @classmethod
    def get_param_info(cls):
        """
        Returns structured information about the parameters in this function.

        Inputs:
        - None

        Outputs:
        - params_info (dict): dict in the format {"param_name":{"required":, "default":, "type":, "units"}}. This will contain all the parameter info for a given EEGFunction subclass instantiation
        """


class AllFunctions:
    """
    Registry class that stores all the EEGFunctions we've coded thus far. This will be useful for generating a menu of function options based on the functions we have coded.
    """
    _functions = [] #List of EEGFunction subclasses
    
    @classmethod
    def add_to_functions(cls, Function):
        """
        Function to add a given EEGFunction subclass (e.g., AddNoise) to _functions

        Inputs:
        - Function (EEGFunction subclass): EEGFunction subclass to add

        Outputs:
        None.
        """
        cls._functions.append(Function)
    
    @classmethod
    def get_fxn_names(cls):
        return [fxn.name for fxn in cls._functions]

"""
class example_function(EEGFunction): #change example_function name
    name = 'Function name' #Change Function name to appropriate name 
    params_units_dict = {"arg_1":"units", "arg_2":"unit"} #replace with units

    def __init__(self, arg_1 =None, arg_2 =None, **kwargs): #add the arguments/parameters you want to
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')}
        super().__init__(**params, **kwargs)
        self.__dict__.update(params)
    def _apply_function(self, original_signal, eeg_object, **kwargs):
        #original_signal will contain a list of data points WITHOUT time data
        #access arguments/parameters as self.arg_1
        #add logic
        #return the edited signal datapoints as a list ONLY
        pass
"""

class add_power(EEGFunction):
    name = "Add power"
    params_units_dict = {"frequency":"Hz", "amplitude":"uV", "phase":"radians", "final_amplitude":"uV"}

    def __init__(self, frequency =None, amplitude =None, phase: float = 0, final_amplitude = None, stdev = None, **kwargs):
        """"
        Initializes the add_power function.

        Inputs:
        - frequency (float): frequency of oscillation to add (Hz)
        - amplitude (float): amplitude of oscillation to add (uV)
        - phase (float): phase offset (radians)
        - final_amplitude (float): final amplitude of oscillation to add if you want amplitude to vary linearly over time
        - stdev (float): if not None, this will introduce randomness into the amplitude at each point in line with STDEV
        - **additional parameters passed to parent
        """
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')}
        super().__init__(**params, **kwargs)
        self.__dict__.update(params)

        if self.frequency is None or self.amplitude is None:
            raise ValueError("frequency and amplitude are required")
    
    def _apply_function(self, original_signal, eeg_object, **kwargs):
        """
        Add oscillatory power to a signal.

        Inputs:
        - original_signal (numpy array): signal segment to modify
        - **kwargs: other signals
        """
        srate = eeg_object.srate
        n_samples = len(original_signal)
        t = np.arange(n_samples) / srate

        # Handle amplitude variation if final_amplitude specified
        if self.final_amplitude is not None:
            # Linear amplitude increase from amplitude to final_amplitude
            amplitudes = np.linspace(self.amplitude, self.final_amplitude, n_samples)
        else:
            amplitudes = np.full(n_samples, self.amplitude)

        # Add random variation if requested
        if self.stdev is not None:
            noise = np.random.normal(0, self.stdev, n_samples)
            amplitudes = amplitudes + noise

        # Generate oscillation
        omega = 2 * np.pi * self.frequency
        oscillation = amplitudes * np.sin(omega * t + self.phase)
        
        # Add to original signal
        modified_signal = original_signal + oscillation
        return modified_signal

class art_reject(EEGFunction):
    name = "Artefact Rejection" #Required
    params_units_dict = {"max_zero_length": "secs", "high_amp_collar": "secs", "jump_collar": "secs", "max_repeat_length":"secs", "max_voltage":"uV", "max_jump":"uV"} #Required
    
    
    def __init__(self, max_zero_length=1, high_amp_collar=10, jump_collar=0.5, max_repeat_length=0.1, max_voltage=1500, max_jump=200, **kwargs):
        """
        Initializes the function.
        
        Inputs:
        - max_zero_length: Number of seconds above which a run of 0s is considered artefact
        - high_amp_collar: Number of seconds rejected before and after a high-ampitude artefact 
        - jump_collar: Number of seconds rejected before and after sudden-jump and repeated-values artefacts
        - max_repeat_length: Number of seconds above which a run of any value is considered artefact
        - max_voltage: Maximum voltage (in uV) allowed before signal is considered artefact 
        - max_jump: Maximum voltage difference (in uV) allowed between consecutive values before signal is considered artefact
        """
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')} #Leave unchanged
        super().__init__(**params, **kwargs) #Leave unchanged
        self.__dict__.update(params) #Leave unchanged

        #Edit quality checks
        if self.max_voltage is None or self.max_voltage <= 0:
            raise ValueError("max_voltage is required and must be positive")
        if self.max_jump is None or self.max_jump <= 0:
            raise ValueError("max_jump is required and must be positive")
        if self.max_zero_length is None or self.max_zero_length <= 0:
            raise ValueError("max_zero_length is required and must be positive")
        if self.max_repeat_length is None or self.max_repeat_length <= 0:
            raise ValueError("max_repeat_length is required and must be positive")
        if self.high_amp_collar is None or self.high_amp_collar < 0:
            raise ValueError("high_amp_collar is required and must be non-negative")
        if self.jump_collar is None or self.jump_collar < 0:
            raise ValueError("jump_collar is required and must be non-negative")
    
    def _apply_function(self, original_signal, eeg_object, **kwargs):
        """
        Apply NEURAL_py single-channel artefact rejection.
        
        Inputs:
        - original_signal (numpy array): signal segment to modify
        - **kwargs: other parameters
        
        Output:
        - modified_signal (numpy array): filtered signal
        """
        params = {
        'max_zero_length': self.max_zero_length,
        'high_amp_collar': self.high_amp_collar,
        'jump_collar': self.jump_collar,
        'max_repeat_length': self.max_repeat_length,
        'max_voltage': self.max_voltage,
        'max_jump': self.max_jump
        }
        modified_signal, amount_removed = art_per_channel(original_signal, eeg_object.srate, params)
        return modified_signal
    
class bandpass_filter(EEGFunction):
    name = "Bandpass Filter" #Required
    params_units_dict = {"lowpass": "Hz", "highpass": "Hz"} #Required
    
    def __init__(self, lowpass=None, highpass=None, order: int = 4, **kwargs):
        """
        Initializes the bandpass_filter function.
        
        Inputs:
        - lowpass (float): low frequency cutoff for the bandpass filter (Hz)
        - highpass (float): high frequency cutoff for the bandpass filter (Hz)
        - order (int): filter order (default 4)
        - **kwargs: additional parameters passed to parent
        """
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')} #Leave unchanged
        super().__init__(**params, **kwargs) #Leave unchanged
        self.__dict__.update(params) #Leave unchanged

        #Edit quality checks
        if self.lowpass is None or self.highpass is None:
            raise ValueError("lowpass and highpass are required")
        if self.lowpass >= self.highpass:
            raise ValueError("lowpass must be less than highpass")
    
    def _apply_function(self, original_signal, eeg_object, **kwargs): #Edit
        """
        Apply bandpass filter to a signal using a Butterworth filter.
        
        Inputs:
        - original_signal (numpy array): signal segment to modify
        - **kwargs: other parameters
        
        Output:
        - modified_signal (numpy array): filtered signal
        """
        # Design the Butterworth bandpass filter
        nyquist = eeg_object.srate / 2
        low = self.lowpass / nyquist
        high = self.highpass / nyquist
        if self.highpass >= eeg_object.srate / 2:
            raise ValueError("highpass must be less than Nyquist frequency (srate/2)")
        sos = butter(self.order, [low, high], btype='band', output='sos')
        # Apply the filter using zero-phase filtering
        modified_signal = sosfiltfilt(sos, original_signal)
        print("Bandpass filtering done!")
        return modified_signal
    
class NotchFilter(EEGFunction):
    name = "Notch filter"
    params_units_dict = {"q_factor":"unitless", "notch_frequency":"Hz"}

    def __init__(self, notch_frequency=60, q_factor=50, **kwargs):
        """
        Initializes the notch_filter function.
        
        Inputs:
        - notch_frequency (float): frequency we would like to filter out (Hz)
        - q_factor (float): factor describing the tightness of the band around the frequency we filter out (unitless)
        - **kwargs: additional parameters passed to parent
        """
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')} #Leave unchanged
        super().__init__(**params, **kwargs) #Leave unchanged
        self.__dict__.update(params) #Leave unchanged

        #Edit quality checks
        if self.q_factor is None:
            raise ValueError("q_factor is required; it is currently None")
        elif self.notch_frequency is None:
            raise ValueError("notch_frequency is required; it is currently None")
        elif self.notch_frequency<0:
            raise ValueError(f"notch_frequency must be greater than 0; yours is {self.notch_frequency}")
    
    def notch_filter_signal(my_raw):
    """
    Code to create and apply a notch filter to the my_raw object of interest.
    
    Inputs:
    - my_raw (alex_raw object): object whose data we want to notch filter

    Outputs:
    - my_raw (alex_raw object): returns the alex_raw object with the updated EEG_data array
    """
    notch_frequency = vars_dict["notch_frequency"]
    q_factor = vars_dict["notch_q"]
    
    def iir_notch_filter(data, notch_freq, quality_factor, fs):
        b, a = iirnotch(notch_freq / (fs / 2), quality_factor)
        return filtfilt(b, a, data)
    
    data_init = np.copy(my_raw.EEG_data)
    time_to_pad = 1/notch_frequency*3
    n_to_pad = int(time_to_pad*my_raw.sampling_rate)
    
    for start, length in zip(my_raw.clean_runs, my_raw.clean_runs_lengths):
        data_to_filter = my_raw.EEG_data[start:start+length]
        data_padded = np.pad(data_to_filter, n_to_pad, mode='reflect')
        filtered_segment = iir_notch_filter(data=data_padded, notch_freq=notch_frequency, 
                                         quality_factor=q_factor, fs=my_raw.sampling_rate)
        filtered_segment = filtered_segment[n_to_pad:-n_to_pad]
        if len(data_to_filter) != len(filtered_segment):
            raise ValueError("The length of the EEG data segment and the filtered segment should be the same.")
        my_raw.EEG_data[start:start+length] = filtered_segment

    if np.array_equal(my_raw.EEG_data, data_init):
        raise ValueError("After notch filtering, there was no change to the data.")
    
    return my_raw

class add_artefact(EEGFunction):
    name = "Add noise"
    params_units_dict = {"num_artefacts":"artefacts", "proportion_artefacts":"", "artefact_type":""}

    def __init__(self, num_artefacts: int = 1, proportion_artefacts: float = 0.5, artefact_type: str = None, **kwargs):
        """
        Adds synthetic artefacts to the signal.

        Inputs:
        - num_artefacts (int): the number of artefacts of a given type to add to a signal
        - proportion_artefacts (float): the proportion of the signal to make into artefact
        - artefact_type (str): the type of artefact to add. Currently supported are: ["broadband"]. Defaults to broadband electrical noise.
        """
    
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')} #Leave unchanged
        super().__init__(**params, **kwargs) #Leave unchanged
        self.__dict__.update(params) #Leave unchanged

    def _apply_function(self, original_signal, eeg_object, **kwargs):
        """
        Apply bandpass filter to a signal using a Butterworth filter.
        
        Inputs:
        - original_signal (numpy array): signal segment to modify
        - eeg_object (EEGSignal object): the eeg)signal object from which the signal came
        - **kwargs: other parameters
        
        Output:
        - modified_signal (numpy array): filtered signal
        """
        artefact_options = ["broadband"]
        if self.artefact_type is None or self.artefact_type not in artefact_options:
            self.artefact_type = "broadband"
        if self.artefact_type == "broadband":
            #add self.num_artefacts broadband electrical artefacts, totalling self.proportion_artefacts proportion of the signal
            srate = eeg_object.srate
            n_samples = len(original_signal)
            modified_signal = original_signal.copy()
            # Determine total artefact duration
            if self.proportion_artefacts is not None:
                total_artefact_samples = int(n_samples * self.proportion_artefacts)
            else:
                total_artefact_samples = n_samples // 2  # Default to 50%
            # Divide into num_artefacts chunks
            artefact_duration_samples = total_artefact_samples // self.num_artefacts
            
            for i in range(self.num_artefacts):
                # Random start position
                start_idx = np.random.randint(0, n_samples - artefact_duration_samples)
                end_idx = start_idx + artefact_duration_samples
                
                # Generate broadband noise (50-500 Hz range)
                t = np.arange(artefact_duration_samples) / srate
                noise = np.zeros(artefact_duration_samples)
                for freq in np.arange(50, 500, 10):
                    amplitude = np.random.uniform(50, 200)  # High amplitude noise
                    phase = np.random.uniform(0, 2*np.pi)
                    noise += amplitude * np.sin(2 * np.pi * freq * t + phase)
                
                # Add noise to signal
                modified_signal[start_idx:end_idx] += noise
            
            return modified_signal

class save_image(EEGFunction):
    name = "Save Image"  # Required
    params_units_dict = {"save_path": "full path"}  # Required
    
    def __init__(self, save_path=None, zoom_n: float = None, ext: str = ".png", dpi: int = 300, figsize: tuple = (12, 4), **kwargs):
        """
        Saves an image of the signal to a specified path.
        
        Inputs:
        - save_path (str): the path where we would like the image of the signal saved
        - zoom_n (float): [FUNCTIONALITY NOT ADDED YET] if not none, will zoom in on the first zoom_n seconds of the signal
        - ext (str): image extension. .png by default
        - dpi (int): resolution of saved image (default: 300)
        - figsize (tuple): figure size as (width, height) in inches (default: (12, 4))
        - **kwargs: additional parameters passed to parent
        """
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')}  # Leave unchanged
        super().__init__(**params, **kwargs)  # Leave unchanged
        self.__dict__.update(params)  # Leave unchanged

        if isinstance(self.figsize, str):
            # Parse string like "(12, 4)" to tuple (12, 4)
            self.figsize = tuple(map(float, self.figsize.strip('()').split(',')))
        elif self.figsize is not None:
            # Ensure it's a tuple of numeric types
            self.figsize = tuple(float(x) for x in self.figsize)
    
    def _apply_function(self, original_signal, eeg_object, **kwargs):  # Leave this line as is
        """
        Save an image of the signal. NB: TimeSeries will NOT be saved with the signal Figure.
        
        Inputs:
        - original_signal (numpy array): signal segment to modify (not really used in this function)
        - eeg_object (EEGSignal instance): EEGSignal whose signal we would like to save
        - **kwargs: other parameters
        
        Output:
        - modified_signal (numpy array): original signal (unchanged, for compatibility with .apply() architecture)
        """
        name = eeg_object.name
        
        # Ensure save directory exists
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct filepath
        save_path = save_dir / f"{name}{self.ext}"
        
        # Only create and save if file doesn't already exist
        if not save_path.exists():
            # Get EEG data
            EEG_data = eeg_object.data
            EEG_times = eeg_object.times
            
            # Data integrity check
            if not(len(EEG_data) == len(EEG_times)):
                raise ValueError(f"The number of data points and time points should be the same; however, you have {len(EEG_data)} data points and {len(EEG_times)} time points.")
            
            # Downsample helper function
            def downsample_data(times, data, max_points=1000):
                if len(data) <= max_points:
                    return times, data
                
                factor = max(1, len(data) // (max_points // 2))
                downsampled_times = []
                downsampled_data = []
                
                for i in range(0, len(data), factor):
                    window_end = min(i + factor, len(data))
                    window_data = data[i:window_end]
                    window_times = times[i:window_end]
                    
                    if len(window_data) > 0:
                        min_idx = np.argmin(window_data)
                        max_idx = np.argmax(window_data)
                        
                        if min_idx < max_idx:
                            downsampled_times.extend([window_times[min_idx], window_times[max_idx]])
                            downsampled_data.extend([window_data[min_idx], window_data[max_idx]])
                        else:
                            downsampled_times.extend([window_times[max_idx], window_times[min_idx]])
                            downsampled_data.extend([window_data[max_idx], window_data[min_idx]])
                
                return downsampled_times, downsampled_data
            
            # Downsample if needed
            plot_times, plot_data = downsample_data(EEG_times, EEG_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot EEG signal
            ax.plot(plot_times, plot_data, linewidth=0.5)
            ax.set_ylabel('Amplitude (uV)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'EEG Signal: {name}')
            ax.grid(True)
            ax.set_xlim(min(EEG_times), max(EEG_times))
            
            # Format y-axis to avoid scientific notation
            from matplotlib.ticker import ScalarFormatter
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            
            # Plot flags if they exist
            if hasattr(eeg_object, 'flags') and len(eeg_object.flags) > 0:
                flags = eeg_object.flags
                color_idx = 1
                for flag_name, flag_value in flags.items():
                    color = f'C{color_idx}'
                    color_idx += 1
                    if len(flag_value) == 1:
                        ax.axvline(x=flag_value[0], color=color, linestyle='--', linewidth=1.5, label=flag_name)
                    elif len(flag_value) == 3:
                        ax.axvline(x=flag_value[0], color=color, linestyle='--', linewidth=1.5)
                        ax.axvline(x=flag_value[1], color=color, linestyle='--', linewidth=1.5, label=flag_name)
                        if flag_value[2]:
                            ax.axvspan(flag_value[0], flag_value[1], alpha=0.2, color=color)
                ax.legend(loc='upper right', fontsize='small')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Signal {name} saved to {save_path}!")
        else:
            print(f"Signal {name} already exists at {save_path}, skipping save.")
        
        return original_signal  # Returns original signal to maintain compatibility with .apply() architecture