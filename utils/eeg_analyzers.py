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
        # Only register if 'name' is a concrete string (not an abstract property)
        if isinstance(cls.__dict__.get("name"), str):
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
        print(f"[Info from _iterate_windows in eeg_analyzers.py] data_len={len(analyze_data)}, "
            f"n_window={n_window_samples}, n_step={n_step_samples}, "
            f"clean_segments type={type(clean_segments)}, "
            f"clean_segments={'None' if clean_segments is None else f'len={len(clean_segments)}, segments={clean_segments[:5]}'}")
        if clean_segments is None:
            for start_i in range(0, len(analyze_data) - n_window_samples + 1, n_step_samples):
                end_i = start_i + n_window_samples
                window_signal = analyze_data[start_i:end_i]
                if np.any(np.isnan(window_signal)):
                    continue
                mid_i = start_i + n_window_samples // 2
                yield start_i, end_i, window_signal, analyze_times[mid_i]
        else:
            for seg_start, seg_end in clean_segments:
                for start_i in range(seg_start, seg_end - n_window_samples + 1, n_step_samples):
                    end_i = start_i + n_window_samples
                    if end_i > seg_end:
                        break
                    window_signal = analyze_data[start_i:end_i]
                    if np.any(np.isnan(window_signal)):
                        continue
                    mid_i = start_i + n_window_samples // 2
                    yield start_i, end_i, window_signal, analyze_times[mid_i]


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

    @classmethod
    def get_params(cls, eeg_object=None, parent=None):
        """
        Collects any required instantiation parameters from the user via GUI.
        Base implementation returns empty dict (no extra params needed).
        Subclasses (e.g., MultiChannelAnalyzer) override as needed.

        Inputs:
        - eeg_object (EEGSignal or None): the signal, used to offer channel choices etc.
        - parent (tkinter widget or None): parent window for dialogues

        Outputs:
        - dict of kwargs to pass to __init__, or None if the user cancelled.
        """
        return {}

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
            "srate": 1 / self.time_details["advance_time"],
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

    def __init__(self, channels=None, required_num_channels="", **kwargs):
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

    @classmethod
    def get_params(cls, eeg_object=None, parent=None):
        """
        Prompts the user to select which channels this multi-channel analyzer
        should operate on, following the same pattern as EEGFunction.get_params.

        Inputs:
        - eeg_object (EEGSignal or None): signal whose channels are available for selection
        - parent (tkinter widget or None): parent window for dialogues

        Outputs:
        - dict: {"channels": [list of selected channel name strings]}, or None if cancelled
        """
        from eeg_theme_park.utils.gui_utilities import dropdown_menu, simple_dialogue, text_entry

        # Determine required number of channels from a temporary instance
        temp = cls.__new__(cls)
        # Pull required_num_channels from the class's __init__ defaults
        import inspect
        sig = inspect.signature(cls.__init__)
        req_param = sig.parameters.get('required_num_channels', None)
        if req_param is not None and req_param.default is not inspect.Parameter.empty:
            required = req_param.default
        else:
            required = ""  # fallback default

        # If "all", no need to prompt
        if required == "all":
            if eeg_object is not None:
                return {"channels": list(eeg_object.all_channel_labels)}
            else:
                return {"channels": []}

        # Build prompt text
        if isinstance(required, int):
            prompt = f"Select exactly {required} channel(s) for '{cls.name}':"
        else:
            prompt = f"Select channels for '{cls.name}':"

        # If we have the signal, use dropdown; otherwise fall back to text entry
        if eeg_object is not None:
            available = eeg_object.all_channel_labels
            selected = dropdown_menu(prompt, available, multiple=True, parent=parent)
        else:
            raw = text_entry(
                f"{prompt}\nEnter channel names separated by commas:",
                parent=parent
            )
            if raw is None:
                return None
            selected = [ch.strip() for ch in raw.split(",") if ch.strip()]

        if selected is None or len(selected) == 0:
            return None

        # Validate count
        if isinstance(required, int) and len(selected) != required:
            simple_dialogue(
                f"'{cls.name}' requires exactly {required} channel(s), "
                f"but {len(selected)} were selected. Please try again."
            )
            return None

        return {"channels": selected}
    
    def _validate_channels(self, eeg_object):
        """Validate that self.channels is properly set and channels exist."""
        if len(self.channels) == 0:
            raise ValueError(
                f"Analyzer '{self.name}' requires channels to be specified before apply(). "
                f"Available channels: {eeg_object.all_channel_labels}"
            )
        
        # Resolve "all". Note that previously passed channels will be retained here
        if self.required_num_channels == "all" and len(self.channels) == 0:
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
            "srate": 1 / self.time_details["advance_time"],
            # Multi-channel analyzers produce a single TimeSeries, no channel_data
        }

        new_ts = TimeSeries(**params_dict)
        eeg_object.time_series.append(new_ts)
        return eeg_object
    
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

class InterChannelDifference(MultiChannelAnalyzer):
    """
    A basic multi-channel analyzer that computes the mean absolute difference
    between two channels at each time point within a window.
    
    For a window of N samples from channels A and B, this returns:
        mean(|A[i] - B[i]|) for i in 0..N-1
    
    This gives a single scalar per window summarizing how much the two
    channels diverge on average during that period.
    """
    name = "inter_channel_difference"
    units = "µV"  # assuming EEG amplitude units; adjust if needed
    time_details = {"window_length":30, "advance_time":15}

    def __init__(self, channels=None, **kwargs):
        super().__init__(channels=channels, required_num_channels=2, **kwargs)

    def _apply_function_multi(self, window_signals, eeg_object, **kwargs):
        """
        Inputs:
        - window_signals (np.array): shape (2, n_window_samples)
        - eeg_object (EEGSignal): parent signal
        
        Returns:
        - float: mean absolute difference between the two channels in this window
        """
        return float(np.mean(np.abs(window_signals[0] - window_signals[1])))

class AveragePower(MultiChannelAnalyzer):
    """
    This mirrors the 'rdp' feature from wrt_calcFeatures.m:
        allchsavgS = mean([S_ch1, S_ch2, ...], 2)
        rdp = mean(allchsavgS(f>1 & f<4)) / mean(allchsavgS)
    
    Chronux-equivalent parameters:
        tapers = [3 5]  →  NW=3, K=5 DPSS tapers
        fpass  = [1 40] Hz
        pad    = -1     →  no zero-padding (NFFT = window length)
    """
    name = "average_power"
    units = "ratio"
    time_details = {"window_length": 30, "advance_time": 15}

    # Spectral parameters (Chronux equivalents)
    NW = 3          # time-bandwidth product
    K = 5           # number of tapers
    fpass = (1, 40) # frequency range of interest in Hz

    def __init__(self, channels=None, **kwargs):
        super().__init__(channels=channels, required_num_channels="all", **kwargs)

    def _apply_function_multi(self, window_signals, eeg_object, **kwargs):
        """
        Inputs:
        - window_signals (np.array): shape (n_channels, n_window_samples)
        - eeg_object (EEGSignal): parent signal (for srate)

        Returns:
        - float: mean power in 1–4 Hz / mean power in 1–40 Hz, averaged across channels
        """
        srate = eeg_object.srate
        n_channels, n_samples = window_signals.shape
        nfft = n_samples  # pad = -1 in Chronux means no zero-padding

        # Generate DPSS tapers (equivalent to Chronux [NW, K])
        tapers = dpss(n_samples, self.NW, self.K)  # shape: (K, n_samples)

        # Frequency vector
        freqs = np.fft.rfftfreq(nfft, d=1.0 / srate)

        # Mask for fpass and total band
        fpass_mask = (freqs >= self.fpass[0]) & (freqs <= self.fpass[1])
        total_mask = (freqs > 1) & (freqs < 40)

        # Compute multitaper spectrum per channel, then average across channels
        # This mirrors: for ch=1:nchs → mtspectrumc → end; allchsavgS = mean(allchsS, 2)
        all_channels_S = np.zeros(np.sum(fpass_mask))

        for ch in range(n_channels):
            # Multitaper estimate: average |FFT(taper * signal)|^2 across tapers
            ch_S = np.zeros(len(freqs))
            for k in range(self.K):
                tapered = tapers[k, :] * window_signals[ch, :]
                spectrum = np.abs(np.fft.rfft(tapered, n=nfft)) ** 2
                ch_S += spectrum
            ch_S /= self.K  # average across tapers
            all_channels_S += ch_S[fpass_mask]

        all_channels_S /= n_channels  # average across channels

        # Relative delta power: mean(S in delta) / mean(S in fpass)
        freqs_in_fpass = freqs[fpass_mask]
        total_in_fpass = (freqs_in_fpass > 1) & (freqs_in_fpass < 40)

        rdp = np.mean(all_channels_S[total_in_fpass])
        return float(rdp)
    
class AveragePowerSimple(MultiChannelAnalyzer):
    """
    Mean spectral power in 1–40 Hz, averaged across all channels.
    Uses Welch's method for simplicity.
    """
    name = "average_power_simple"
    units = "uV^2/Hz"
    time_details = {"window_length": 30, "advance_time": 15}

    def __init__(self, channels=None, **kwargs):
        super().__init__(channels=channels, required_num_channels="all", **kwargs)

    def _apply_function_multi(self, window_signals, eeg_object, **kwargs):
        from scipy.signal import welch

        srate = eeg_object.srate
        n_channels = window_signals.shape[0]
        ch_powers = np.zeros(n_channels)

        for ch in range(n_channels):
            freqs, psd = welch(window_signals[ch, :], fs=srate, nperseg=min(window_signals.shape[1], 2 * int(srate)), return_onesided=False)
            mask = (freqs >= 1) & (freqs <= 40)
            ch_powers[ch] = np.mean(psd[mask])

        return float(np.mean(ch_powers))
    
class DeltaPowerSimple(MultiChannelAnalyzer):
    """
    Mean spectral power in 1–40 Hz, averaged across all channels.
    Uses Welch's method for simplicity.
    """
    name = "delta_power_simple"
    units = "uV^2/Hz"
    time_details = {"window_length": 30, "advance_time": 15}

    def __init__(self, channels=None, **kwargs):
        super().__init__(channels=channels, required_num_channels="all", **kwargs)

    def _apply_function_multi(self, window_signals, eeg_object, **kwargs):
        from scipy.signal import welch

        srate = eeg_object.srate
        n_channels = window_signals.shape[0]
        ch_powers = np.zeros(n_channels)

        for ch in range(n_channels):
            freqs, psd = welch(window_signals[ch, :], fs=srate, nperseg=min(window_signals.shape[1], 2 * int(srate)), return_onesided=False)
            mask = (freqs >= 1) & (freqs <= 4)
            ch_powers[ch] = np.mean(psd[mask])

        return float(np.mean(ch_powers))
    
class RelativeDeltaPowerSimple(MultiChannelAnalyzer):
    name = "rdp_simple"
    units = "uV^2/Hz"

    time_details = {"window_length": 30, "advance_time": 15}

    def __init__(self, channels=None, **kwargs):
        super().__init__(channels=channels, required_num_channels="all", **kwargs)
    
    def _apply_function_multi(self, window_signals, eeg_object, **kwargs):
        from scipy.signal import welch

        srate = eeg_object.srate
        n_channels = window_signals.shape[0]
        delta_ch_powers = np.zeros(n_channels)

        #Get delta power
        for ch in range(n_channels):
            freqs, psd = welch(window_signals[ch, :], fs=srate, nperseg=min(window_signals.shape[1], 2 * int(srate)), return_onesided=False)
            mask = (freqs >= 1) & (freqs <= 4)
            delta_ch_powers[ch] = np.mean(psd[mask])
        delta_power = float(np.mean(delta_ch_powers))

        #Get total power
        total_ch_powers = np.zeros(n_channels)
        for ch in range(n_channels):
            freqs, psd = welch(window_signals[ch, :], fs=srate, nperseg=min(window_signals.shape[1], 2 * int(srate)), return_onesided=False)
            mask = (freqs >= 1) & (freqs <= 40)
            total_ch_powers[ch] = np.mean(psd[mask])

        total_power = float(np.mean(total_ch_powers))

        return delta_power/total_power