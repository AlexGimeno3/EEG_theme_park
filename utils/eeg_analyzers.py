"""
This file will contain the different functions that we want to use to analyze our signals.
"""

from abc import ABC, abstractmethod
from eeg_theme_park.utils.eeg_signal import TimeSeries
from eeg_theme_park.utils.NEURAL_py_fork.spectral_features import main_spectral
import numpy as np
from tqdm import tqdm
import copy

class EEGAnalyzer(ABC):
    """
    Abstract class for EEG analysis functions.
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
    def units(self):
        """
        Each subclass must define units. If function is unitless, can be left as "".
        """
        pass

    @property
    @abstractmethod
    def time_details(self):
        """
        Each function is applied to smaller data segments (i.e., 2-sec windows in 60-sec data). time_details will be a dictionary for each function that specifies the window_length (2 seconds) and the advance_time (i.e., the amount that the window advances; for example, if set to 1 second, windows will be from 0-2 sec, then 1-3 sec, etc)
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
        #Ensure that, for every EEGAnalyzer subclass that is defined below, the subclass is immediately registered in AllAnalyzers and therefore accessible via AllAnalyzers._functions
        super().__init_subclass__(**kwargs)
        AllAnalyzers.add_to_analyzers(cls)
    
    def apply(self, eeg_object, keep_fxn = True):
        """
        Concrete method that 
        Inputs:
        - eeg_object (EEGSignal instance): the EEGSignal instance we are analyzing
        - keep_fxn (boolean): if true, will keep the instance of the instantiated subclass saved in the generated TimeSeries object; this can allow for better reproducability but is a bit harder on memory
        
        Outputs:
        - eeg_object (EEGSignal instance): EEGObject after a TimeSeries has been added
        """
        if self.name in [ts.name for ts in eeg_object.time_series]:
            return eeg_object  # Already computed
        
        #Initialize
        srate = eeg_object.srate
        n_window_samples = int(self.time_details["window_length"]*srate) #Leave untouched
        n_step_samples = int(self.time_details["advance_time"]*srate) #Leave untouched
        ts_values = []
        ts_times = []
        #Get appropriate time range to analyze
        if len(eeg_object.analyze_time_lims) > 0:
            start_time_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[0])
            end_time_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[1])
            analyze_data = eeg_object.data[start_time_i:end_time_i]
            analyze_times = eeg_object.times[start_time_i:end_time_i]
        else:
            analyze_data = eeg_object.data
            analyze_times = eeg_object.times
        #Run our analysis
        for start_i in tqdm(range(0,len(analyze_data)-n_window_samples+1,n_step_samples)):
            end_i = start_i + n_window_samples
            window_signal = analyze_data[start_i:end_i] #NB: this is just a square window
            window_val = self._apply_function(window_signal,eeg_object)
            ts_values.append(window_val)
            ts_times.append(analyze_times[start_i]) #Measurement at start of the window
        
        params_dict = {
            "name" : self.name,
            "values" : ts_values,
            "units" : self.units,
            "times" : ts_times,
            "function" : self if keep_fxn else None
        }

        new_ts = TimeSeries(**params_dict)
        eeg_object.time_series.append(new_ts)
        return eeg_object


    @abstractmethod
    def _apply_function(self, lims, eeg_object, **kwargs):
        """
        Abstract method that must be used when creating a new function. Sublclasses should contain all the code extracts the timeseries.

        Inputs:
        - lims (tuple of int): indices of the current window being analyzed in the form [a, b]
        - eeg_object (EEGSignal object): EEGSignal we will be implementing

        Output:
        - ts_values (list of float): the time series values after the function has been applied
        - ts_times (list of float): the time values (in sec) corresponding to each value in 
        """
        pass

class AllAnalyzers:
    """
    Registry class that stores all the EEGAnalyzers we've coded thus far. This will be useful for generating a menu of function options based on the functions we have coded.
    """
    _analyzers = [] #List of EEGAnalyzer subclasses
    
    @classmethod
    def add_to_analyzers(cls, Analyzer):
        """
        Function to add a given EEGAnalyzer subclass (e.g., DeltaPower) to AllAnalyzers._functions

        Inputs:
        - Analyzer (EEGAnalyzer subclass): EEGAnalyzer subclass to add

        Outputs:
        None.
        """
        cls._analyzers.append(Analyzer)
    
    @classmethod
    def get_analyzer_names(cls):
        return [analyzer.name for analyzer in cls._analyzers]
    
class Power8to10Hz(EEGAnalyzer):
    """"
    Compute power in the 8-10 Hz frequency band
    """
    name = "power_8_10_hz" #Edit to include the correct name
    units = "uV^2" #Edit to include the correct units
    time_details = {"window_length":2, "advance_time":1} #Edit to perform windowing as you would like (currently, this means the function is applied to 2-second-long windows, advancing each time by 1 second [i.e., with 50% overlap])
    
    def __init__(self, **kwargs): #Leave unchanged
        super().__init__(**kwargs) #Leave unchanged
    
    def _apply_function(self, window_signal, eeg_object, **kwargs): #Leave unchanged
            #Analysis code here
            #---------------------------------------
            srate = eeg_object.srate
            freq_low = 8
            freq_high = 10
            fft_values = np.fft.fft(window_signal)
            fft_freqs = np.fft.fftfreq(len(window_signal), 1/srate)
            # Get positive frequencies only
            pos_mask = fft_freqs >= 0
            fft_freqs = fft_freqs[pos_mask]
            fft_values = fft_values[pos_mask]
            # Find indices in frequency band
            freq_mask = (fft_freqs >= freq_low) & (fft_freqs <= freq_high)
            # Compute power (magnitude squared, normalized)
            power_spectrum = np.abs(fft_values) ** 2 / len(window_signal)
            band_power = np.sum(power_spectrum[freq_mask])
            return band_power
            #----------------------

class EdgeFrequency(EEGAnalyzer):
    """
    Compute spectral edge frequency.
    """
    name = "spectral_edge_frequency" #Edit to include the correct name
    units = "Hz" #Edit to include the correct units
    time_details = {"window_length":30, "advance_time":15} #Edit to perform windowing as you would like (currently, this means the function is applied to 2-second-long windows, advancing each time by 1 second [i.e., with 50% overlap])
    
    def __init__(self, **kwargs): #Leave unchanged
        super().__init__(**kwargs) #Leave unchanged
    
    def _apply_function(self, window_signal, eeg_object, **kwargs): #Leave unchanged
            #Analysis code here
            #---------------------------------------
            srate = eeg_object.srate
            pass_params = {
                "L_window" : self.time_details["window_length"], 
                "window_type" : "hamm", #Other option is "rect"; Hamming chosen to reduce spectral leakage
                "overlap" : self.time_details["advance_time"]/self.time_details["window_length"]*100,
                "method" : "periodogram"
                }
            sef = main_spectral(window_signal, srate, "spectral_edge_frequency", pass_params)
            return sef

            freq_low = 8
            freq_high = 10
            fft_values = np.fft.fft(window_signal)
            fft_freqs = np.fft.fftfreq(len(window_signal), 1/srate)
            # Get positive frequencies only
            pos_mask = fft_freqs >= 0
            fft_freqs = fft_freqs[pos_mask]
            fft_values = fft_values[pos_mask]
            # Find indices in frequency band
            freq_mask = (fft_freqs >= freq_low) & (fft_freqs <= freq_high)
            # Compute power (magnitude squared, normalized)
            power_spectrum = np.abs(fft_values) ** 2 / len(window_signal)
            band_power = np.sum(power_spectrum[freq_mask])
            return band_power
            #----------------------