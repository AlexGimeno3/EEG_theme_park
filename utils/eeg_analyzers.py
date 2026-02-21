"""
This file will contain the different functions that we want to use to analyze our signals.
"""

from abc import ABC, abstractmethod
from eeg_theme_park.utils.eeg_signal import TimeSeries
from eeg_theme_park.utils.NEURAL_py_fork.spectral_features import main_spectral, spectral_power
import numpy as np
from tqdm import tqdm
from scipy.signal import hilbert
import antropy as ant
from scipy.signal.windows import dpss
from scipy.signal import hilbert

class EEGAnalyzer(ABC):
    """
    Abstract base class for EEG analysis functions.
    Subclass either SingleChannelAnalyzer or MultiChannelAnalyzer, not this directly.
    """
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def units(self):
        pass

    @property
    @abstractmethod
    def time_details(self):
        pass

    def __init__(self, **kwargs):
        self.args_dict = kwargs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register concrete classes (not SingleChannelAnalyzer/MultiChannelAnalyzer themselves)
        if not getattr(cls, '__abstractmethods__', None):
            AllAnalyzers.add_to_analyzers(cls)

    def _get_analysis_range(self, eeg_object):
        """
        Shared helper: determines the data range and adjusts for analyze_time_lims.
        
        Returns:
        - global_offset (int): index offset into full data array
        - analyze_times (np.array): time values for the analysis window
        - n_window_samples (int): number of samples per window
        - n_step_samples (int): number of samples per step
        """
        srate = eeg_object.srate
        n_window_samples = int(self.time_details["window_length"] * srate)
        n_step_samples = int(self.time_details["advance_time"] * srate)

        if len(eeg_object.analyze_time_lims) > 0:
            start_time_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[0])
            end_time_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[1])
            analyze_times = eeg_object.times[start_time_i:end_time_i]
            global_offset = start_time_i
        else:
            analyze_times = eeg_object.times
            global_offset = 0

        return global_offset, analyze_times, n_window_samples, n_step_samples

    def _adjust_clean_segments(self, clean_segments, global_offset, analyze_length):
        """
        Adjust clean_segments indices relative to the analysis window.
        
        Inputs:
        - clean_segments: list of (start_idx, end_idx) tuples, or None
        - global_offset: start index of the analysis window in full data
        - analyze_length: length of the analysis window
        
        Returns:
        - adjusted segments (list of tuples) or None
        """
        if clean_segments is None:
            return None
        adjusted = []
        for start_idx, end_idx in clean_segments:
            if start_idx < global_offset + analyze_length and end_idx > global_offset:
                adj_start = max(0, start_idx - global_offset)
                adj_end = min(analyze_length, end_idx - global_offset)
                adjusted.append((adj_start, adj_end))
        return adjusted

    def _iterate_windows(self, analyze_data, analyze_times, clean_segments, n_window_samples, n_step_samples):
        """
        Generator yielding (start_i, end_i, window_signal, timestamp) for each valid window.
        """
        if clean_segments is None:
            for start_i in range(0, len(analyze_data) - n_window_samples + 1, n_step_samples):
                end_i = start_i + n_window_samples
                window_signal = analyze_data[start_i:end_i]
                if np.any(np.isnan(window_signal)):
                    continue
                yield start_i, end_i, window_signal, analyze_times[start_i]
        else:
            for seg_start, seg_end in clean_segments:
                for start_i in range(seg_start, seg_end - n_window_samples + 1, n_step_samples):
                    end_i = start_i + n_window_samples
                    if end_i > seg_end:
                        break
                    window_signal = analyze_data[start_i:end_i]
                    if np.any(np.isnan(window_signal)):
                        continue
                    yield start_i, end_i, window_signal, analyze_times[start_i]


class SingleChannelAnalyzer(EEGAnalyzer):
    """
    Base class for analyzers that operate on one channel at a time.
    Loops over all channels in eeg_object.all_channel_data automatically.
    Subclasses must implement _apply_function_single(self, window_signal, eeg_object).
    """
    channel_mode = "single"

    @abstractmethod
    def _apply_function_single(self, window_signal, eeg_object, **kwargs):
        """
        Compute a scalar value from a single window of single-channel data.
        
        Inputs:
        - window_signal (np.array): 1D array of shape (n_window_samples,)
        - eeg_object (EEGSignal): the parent signal (for accessing srate, etc.)
        
        Returns:
        - float: the computed value for this window
        """
        pass

    def apply(self, eeg_object, keep_fxn=True, clean_segments=None):
        """
        Apply analyzer to all channels.
        
        Inputs:
        - eeg_object (EEGSignal instance)
        - keep_fxn (bool): if True, store the analyzer instance in the TimeSeries
        - clean_segments: dict {ch_name: [(start, end), ...]} or flat list or None.
            If a flat list, the same segments are used for all channels.
            If a dict, per-channel segments are used.
        """
        print(f"Calculating {self.name}...")

        if self.name in [ts.name for ts in eeg_object.time_series]:
            return eeg_object

        global_offset, analyze_times, n_window_samples, n_step_samples = self._get_analysis_range(eeg_object)

        channel_data_dict = {}
        original_channel = eeg_object.current_channel  # Save to restore later

        for ch_name in eeg_object.all_channel_labels:
            # Get this channel's data within the analysis range
            ch_full_data = eeg_object.all_channel_data[ch_name]
            if len(eeg_object.analyze_time_lims) > 0:
                start_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[0])
                end_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[1])
                ch_analyze_data = ch_full_data[start_i:end_i]
            else:
                ch_analyze_data = ch_full_data

            # Resolve clean segments for this channel
            if isinstance(clean_segments, dict):
                ch_clean = clean_segments.get(ch_name, None)
            else:
                ch_clean = clean_segments  # flat list or None — shared across channels

            ch_clean_adjusted = self._adjust_clean_segments(ch_clean, global_offset, len(ch_analyze_data))

            ts_values = []
            ts_times = []
            for _, _, window_signal, timestamp in self._iterate_windows(
                ch_analyze_data, analyze_times, ch_clean_adjusted, n_window_samples, n_step_samples
            ):
                window_val = self._apply_function_single(window_signal, eeg_object)
                ts_values.append(window_val)
                ts_times.append(timestamp)

            channel_data_dict[ch_name] = {
                "values": np.asarray(ts_values),
                "times": np.asarray(ts_times),
            }

        eeg_object.current_channel = original_channel  # Restore

        params_dict = {
            "name": self.name,
            "values": channel_data_dict[eeg_object.current_channel]["values"],  # default for _values
            "units": self.units,
            "times": channel_data_dict[eeg_object.current_channel]["times"],    # default for _times
            "function": self if keep_fxn else None,
            "channel_data": channel_data_dict,
            "primary_channel": eeg_object.current_channel,
        }

        new_ts = TimeSeries(**params_dict)
        eeg_object.time_series.append(new_ts)
        return eeg_object


class MultiChannelAnalyzer(EEGAnalyzer):
    """
    Base class for analyzers that operate on multiple channels simultaneously.
    Subclasses must implement _apply_function_multi(self, window_signals, eeg_object)
    where window_signals is a 2D np.array of shape (n_channels, n_window_samples).
    
    Subclasses must set class-level or instance-level:
    - required_num_channels: int, "all", or "" (any number)
    - channels: list of channel name strings (set at __init__ or before apply())
    """

    channel_mode = "multi"

    def __init__(self, channels=None, required_num_channels=2, **kwargs):
        """
        Inputs:
        - channels (list of str or None): channel names to use. If None, must be set before apply().
        - required_num_channels (int, "all", or ""): constraint on how many channels are needed.
        """
        super().__init__(**kwargs)
        self.channels = channels if channels is not None else []
        self.required_num_channels = required_num_channels

    @abstractmethod
    def _apply_function_multi(self, window_signals, eeg_object, **kwargs):
        """
        Compute a scalar value from a single window of multi-channel data.
        
        Inputs:
        - window_signals (np.array): 2D array of shape (n_channels, n_window_samples),
          ordered as specified in self.channels
        - eeg_object (EEGSignal): the parent signal
        
        Returns:
        - float: the computed value for this window
        """
        pass

    def _validate_channels(self, eeg_object):
        """Validate that self.channels is properly set and channels exist."""
        if len(self.channels) == 0:
            raise ValueError(
                f"Analyzer '{self.name}' requires channels to be specified before apply(). "
                f"Available channels: {eeg_object.all_channel_labels}"
            )
        
        # Resolve "all"
        if self.required_num_channels == "all":
            self.channels = list(eeg_object.all_channel_labels)
        
        # Validate count
        if isinstance(self.required_num_channels, int) and len(self.channels) != self.required_num_channels:
            raise ValueError(
                f"Analyzer '{self.name}' requires exactly {self.required_num_channels} channels, "
                f"but {len(self.channels)} were provided: {self.channels}"
            )
        
        # Validate existence
        missing = [ch for ch in self.channels if ch not in eeg_object.all_channel_labels]
        if missing:
            raise ValueError(
                f"Channels {missing} not found in signal. Available: {eeg_object.all_channel_labels}"
            )

    @property
    def display_name(self):
        """Name that includes channel info for unique identification."""
        if self.channels:
            ch_suffix = "_".join(self.channels)
            return f"{self.name}_{ch_suffix}"
        return self.name

    def apply(self, eeg_object, keep_fxn=True, clean_segments=None):
        """
        Apply multi-channel analyzer.
        
        clean_segments: dict {ch_name: [(start, end), ...]} or flat list or None.
            For multi-channel, we use the INTERSECTION of clean segments across
            all specified channels to ensure aligned windows.
        """
        ts_name = self.display_name
        print(f"Calculating {ts_name}...")

        if ts_name in [ts.name for ts in eeg_object.time_series]:
            return eeg_object

        self._validate_channels(eeg_object)

        global_offset, analyze_times, n_window_samples, n_step_samples = self._get_analysis_range(eeg_object)

        # Get per-channel data arrays
        ch_data_arrays = {}
        for ch_name in self.channels:
            ch_full = eeg_object.all_channel_data[ch_name]
            if len(eeg_object.analyze_time_lims) > 0:
                start_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[0])
                end_i = eeg_object.time_to_index(eeg_object.analyze_time_lims[1])
                ch_data_arrays[ch_name] = ch_full[start_i:end_i]
            else:
                ch_data_arrays[ch_name] = ch_full

        analyze_length = len(list(ch_data_arrays.values())[0])

        # For multi-channel: use intersection of clean segments if per-channel
        if isinstance(clean_segments, dict):
            # Compute intersection: a sample is clean only if clean in ALL specified channels
            per_ch_segments = [clean_segments.get(ch, None) for ch in self.channels]
            if any(s is None for s in per_ch_segments):
                merged_clean = None
            else:
                # Build per-sample mask and intersect
                mask = np.ones(analyze_length, dtype=bool)
                for ch_segs in per_ch_segments:
                    ch_mask = np.zeros(analyze_length, dtype=bool)
                    adjusted = self._adjust_clean_segments(ch_segs, global_offset, analyze_length)
                    if adjusted:
                        for s, e in adjusted:
                            ch_mask[s:e] = True
                    mask &= ch_mask
                # Convert mask back to segments
                transitions = np.diff(np.concatenate(([False], mask, [False])).astype(int))
                starts = np.where(transitions == 1)[0]
                ends = np.where(transitions == -1)[0]
                merged_clean = list(zip(starts.tolist(), ends.tolist()))
        else:
            merged_clean = self._adjust_clean_segments(clean_segments, global_offset, analyze_length)

        # Use the first channel's data as the reference for window iteration
        # (all channels have the same length, so window positions are the same)
        ref_data = list(ch_data_arrays.values())[0]

        ts_values = []
        ts_times = []
        for start_i, end_i, _, timestamp in self._iterate_windows(
            ref_data, analyze_times, merged_clean, n_window_samples, n_step_samples
        ):
            # Stack windows from all channels
            window_signals = np.array([
                ch_data_arrays[ch][start_i:end_i] for ch in self.channels
            ])
            # Check for NaN in any channel
            if np.any(np.isnan(window_signals)):
                continue

            window_val = self._apply_function_multi(window_signals, eeg_object)
            ts_values.append(window_val)
            ts_times.append(timestamp)

        params_dict = {
            "name": ts_name,
            "values": ts_values,
            "units": self.units,
            "times": ts_times,
            "function": self if keep_fxn else None,
            # Multi-channel analyzers produce a single TimeSeries, no channel_data
        }

        new_ts = TimeSeries(**params_dict)
        eeg_object.time_series.append(new_ts)
        return eeg_object

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
    
class Power8to10Hz(SingleChannelAnalyzer):
    """"
    Compute power in the 8-10 Hz frequency band
    """
    name = "power_8_10_hz" #Edit to include the correct name
    units = "uV^2" #Edit to include the correct units
    time_details = {"window_length":30, "advance_time":15} #Edit to perform windowing as you would like (currently, this means the function is applied to 30-second-long windows, advancing each time by 15 second [i.e., with 50% overlap])
    
    def __init__(self, **kwargs): #Leave unchanged
        super().__init__(**kwargs) #Leave unchanged
    
    def _apply_function_single(self, window_signal, eeg_object, **kwargs): #Leave unchanged
            #Analysis code here
            #---------------------------------------
            srate = eeg_object.srate
            params = {
                "freq_bands" : [[8,10]],
                "method" : "periodogram",
                "L_window" : 2, 
                "window_type" : "hamm", #Other option is "rect"; Hamming chosen to reduce spectral leakage
                "overlap" : 50,

            }
            return spectral_power(window_signal, srate, "spectral_power",params)[0]
            #----------------------

class EdgeFrequency(SingleChannelAnalyzer):
    """
    Compute spectral edge frequency.
    """
    name = "spectral_edge_frequency" #Edit to include the correct name
    units = "Hz" #Edit to include the correct units
    time_details = {"window_length":30, "advance_time":15} #Edit to perform windowing as you would like (currently, this means the function is applied to 2-second-long windows, advancing each time by 1 second [i.e., with 50% overlap])
    
    def __init__(self, **kwargs): #Leave unchanged
        super().__init__(**kwargs) #Leave unchanged
    
    def _apply_function_single(self, window_signal, eeg_object, **kwargs): #Leave unchanged
            #Analysis code here
            #---------------------------------------
            srate = eeg_object.srate
            pass_params = {
                "L_window" : 2, 
                "window_type" : "hamm", #Other option is "rect"; Hamming chosen to reduce spectral leakage
                "overlap" : 50,
                "method" : "periodogram",
                "SEF" : 0.95, #spectral edge frequency threshold
                "total_freq_bands" : [0.5, 30]  #Frequency range within which we are calculating SEF
                }
            sef = main_spectral(window_signal, srate, "spectral_edge_frequency", pass_params)
            return sef
            #----------------------