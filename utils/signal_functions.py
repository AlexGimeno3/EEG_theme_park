"""
This contains the different functions we may want to apply to our signals. It should be built to handle all kinds of scenarios; filtering, adding noise, adding features. Each function should take in the eeg_signal_obj of interest, as well as the fxn_lims (i.e., the start and end time of the function we would like to apply), flags (i.e., do we want to add flags to where we did this so they are displayed in the viewer?) and **kwargs  
"""

from abc import ABC, abstractmethod
import numpy as np
import inspect
from eeg_theme_park.utils.gui_utilities import simple_dialogue
import tkinter as tk
from tkinter import ttk
import copy
from scipy.signal import butter, filtfilt

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
    
    def apply(self, eeg_object, time_range=None, flags_bool=True, min_clean_length = 0):
        """
        Template method to apply the given function to a signal. Done so certain internal processes are always carried out. Sublclasses should contain all the code that modifies the signal, but does not need to log (this is handled internally by the EEGSignal object).

        Inputs:
        - eeg_object (EEGSignal object): EEGSignal object that 
        - time_range (list): time range in seconds over which to apply the function (NB: this is indexed to the actual time values of this signal, not to zero). If none, defaults to the entire signal
        - flags_bool (bool): if True, will add to the EEGSignal.flags dictionary the flags we need
        - min_clean_length (float): minimum length (in seconds) of clean segments to process. Segments shorter than this will be marked as NaN. 

        Output:
        - eeg_object (EEGSignal object): EEGSignal object after our change has been applied
        """
        if time_range is None:
            time_range = [eeg_object.times[0],eeg_object.times[-1]]
        try: # Get the indices corresponding to the time range; raise exception if an error occurs
            start_idx = eeg_object.time_to_index(time_range[0])
            end_idx = eeg_object.time_to_index(time_range[1])
        except Exception as e:
            raise type(e)(f"{str(e)}; the time range you asked to apply the function to was {time_range[0]}-{time_range[1]} secs, but the signal only goes from {eeg_object.start_time} to {eeg_object.end_time} secs.") from e

        if min_clean_length > 0: #Perform cleaning of too-short segments
                from eeg_theme_park.utils.pipeline import find_clean_segments
                data_to_process = eeg_object.data[start_idx:end_idx+1]
                # Find clean segments within this data
                clean_segments = find_clean_segments(data_to_process, eeg_object.srate, min_clean_length)
                
                # Create output array (initialize with NaN)
                processed_data = np.full_like(data_to_process, np.nan)
                
                # Process each clean segment
                for seg_start, seg_end in clean_segments:
                    segment_data = data_to_process[seg_start:seg_end]
                    processed_segment = self._apply_function(segment_data)
                    processed_data[seg_start:seg_end] = processed_segment
                
                eeg_object.data[start_idx:end_idx+1] = processed_data     
        
        else: # No clean segment filtering
            data_to_process = eeg_object.data[start_idx:end_idx+1]
            edited_data = self._apply_function(data_to_process)
            eeg_object.data[start_idx:end_idx+1] = edited_data
        
        if flags_bool:
            eeg_object.add_flag(self.name, copy.deepcopy(time_range))
        eeg_object.has_unsaved_changes = True
        
        return eeg_object

    
    @abstractmethod
    def _apply_function(self, original_signal, **kwargs):
        """
        Abstract method that must be used when creating a new function. Sublclasses should contain all the code that modifies the signal.

        Inputs:
        - original_signal (list of int): the original signal (in uV) that we would like to modify

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
    def _apply_function(self, original_signal, **kwargs):
        #original_signal will contain a list of data points WITHOUT time data
        #access arguments/parameters as self.arg_1
        #add logic
        #return the edited signal datapoints as a list ONLY
        pass
"""

class add_power(EEGFunction):
    name = "Add power"
    params_units_dict = {"frequency":"Hz", "amplitude":"uV", "phase":"radians", "final_amplitude":"uV"}

    def __init__(self, frequency =None, amplitude =None, phase: float = 0, final_amplitude = None, stdev = None, srate = None, **kwargs):
        """"
        Initializes the add_power function.

        Inputs:
        - frequency (float): frequency of oscillation to add (Hz)
        - amplitude (float): amplitude of oscillation to add (uV)
        - phase (float): phase offset (radians)
        - final_amplitude (float): final amplitude of oscillation to add if you want amplitude to vary linearly over time
        - stdev (float): if not None, this will introduce randomness into the amplitude at each point in line with STDEV
        - srate (float): sampling rate of our signal (Hz)
        - **additional parameters passed to parent
        """
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')}
        super().__init__(**params, **kwargs)
        self.__dict__.update(params)

        if self.frequency is None or self.amplitude is None or self.srate is None:
            raise ValueError("frequency, amplitude, and srate are required")
    
    def _apply_function(self, original_signal, **kwargs):
        """
        Add oscillatory power to a signal.

        Inputs:
        - original_signal (numpy array): signal segment to modify
        - **kwargs: other signals
        """
        srate = self.srate
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
    
    
    def __init__(self, max_zero_length=None, high_amp_collar=None, jump_collar=None, max_repeat_length=None, max_voltage=None, max_jump=None, **kwargs):
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
        if self.lowcut is None or self.highcut is None or self.srate is None:
            raise ValueError("lowcut, highcut, and srate are required")
        if self.lowcut >= self.highcut:
            raise ValueError("lowcut must be less than highcut")
        if self.highcut >= self.srate / 2:
            raise ValueError("highcut must be less than Nyquist frequency (srate/2)")
    
    def _apply_function(self, original_signal, **kwargs): #Edit
        """
        Apply bandpass filter to a signal using a Butterworth filter.
        
        Inputs:
        - original_signal (numpy array): signal segment to modify
        - **kwargs: other parameters
        
        Output:
        - modified_signal (numpy array): filtered signal
        """
        # Design the Butterworth bandpass filter
        nyquist = self.srate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        # Apply the filter using zero-phase filtering
        modified_signal = filtfilt(b, a, original_signal)
        return modified_signal
    
class bandpass_filter(EEGFunction):
    name = "Bandpass Filter" #Required
    params_units_dict = {"lowpass": "Hz", "highpass": "Hz", "srate": "Hz"} #Required
    
    def __init__(self, lowcut=None, highcut=None, srate=None, order: int = 4, **kwargs):
        """
        Initializes the bandpass_filter function.
        
        Inputs:
        - lowpass (float): low frequency cutoff for the bandpass filter (Hz)
        - highpass (float): high frequency cutoff for the bandpass filter (Hz)
        - srate (float): sampling rate of the signal (Hz)
        - order (int): filter order (default 4)
        - **kwargs: additional parameters passed to parent
        """
        params = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', '__class__')} #Leave unchanged
        super().__init__(**params, **kwargs) #Leave unchanged
        self.__dict__.update(params) #Leave unchanged

        #Edit quality checks
        if self.lowcut is None or self.highcut is None or self.srate is None:
            raise ValueError("lowcut, highcut, and srate are required")
        if self.lowcut >= self.highcut:
            raise ValueError("lowcut must be less than highcut")
        if self.highcut >= self.srate / 2:
            raise ValueError("highcut must be less than Nyquist frequency (srate/2)")
    
    def _apply_function(self, original_signal, **kwargs): #Edit
        """
        Apply bandpass filter to a signal using a Butterworth filter.
        
        Inputs:
        - original_signal (numpy array): signal segment to modify
        - **kwargs: other parameters
        
        Output:
        - modified_signal (numpy array): filtered signal
        """
        # Design the Butterworth bandpass filter
        nyquist = self.srate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        # Apply the filter using zero-phase filtering
        modified_signal = filtfilt(b, a, original_signal)
        return modified_signal