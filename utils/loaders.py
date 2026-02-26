"""
Interface-based module containing various loaders for EEG files of different formats.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import re
import datetime as dt
from eeg_theme_park.utils.eeg_signal import EEGSignal
import pickle as pkl
import mne
from mne._fiff import utils
from eeg_theme_park.utils.gui_utilities import choose_channel, yes_no, text_entry
from eeg_theme_park.utils.misc_utils import get_datetime_from_excel
import numpy as np

class EEGLoader(ABC):
    """
    Abstract class.
    """
    def __init_subclass__(cls, **kwargs):
        #Ensure that, for every EEGLoader subclass that is defined below, the subclass is immediately registered in AllLoaders and therefore accessible via AllLoaders.get_supported_extensions()
        super().__init_subclass__(**kwargs)
        AllLoaders.add_to_loaders(cls)
        
    @abstractmethod
    def load(self, file_path: Path, **kwargs) -> EEGSignal:
        """
        Abstract loading method. Must always take a file path, and returns an EEGSignal object.

        Inputs:
        - self_path (Path object): the full path to the file being loaded
        - **kwargs: any parameters specific to the format being loaded

        Outputs:
        - eeg_signal_object (EEGSignal object): EEGSignal object containing the loaded data
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_supported_extensions() -> list:
        """
        Inputs:
        None.

        Outputs:
        - extensions (list of str): list of extensions supported by a given EEGLoader subclass
        """
        pass

    def get_channel(self, channel_names, provided_channel=None, **kwargs):
        """
        Helper method to handle channel selection with auto-select support. All loaders should use this instead of calling choose_channel directly.
        
        Inputs:
        - channel_names (list): Available channel names from the file
        - provided_channel (str or None): Channel explicitly requested
        - **kwargs: May contain 'auto_select_channel'
        
        Outputs:
        - (channel_name, channel_index): Selected channel and its index
        """
        auto_select = kwargs.get('auto_select_channel', False)
        
        # Check if provided channel is valid
        if provided_channel and provided_channel in channel_names:
            return (provided_channel, channel_names.index(provided_channel))
        
        # Need to select a channel - either auto or via GUI
        return choose_channel(channel_names, auto_select=auto_select)

class AllLoaders:
    """
    Registry class that stores all the loaders we have coded thus far.
    """
    _loaders = [] #List of EEGLoader subclasses

    @classmethod
    def add_to_loaders(cls, Loader):
        """
        Function to add a given EEGLoader subclass (e.g., EDFLoader) to _loaders

        Inputs:
        - Loader (EEGLoader subclass): EEGLoader subclass to add

        Outputs:
        None.
        """
        cls._loaders.append(Loader)
    
    @classmethod
    def get_supported_extensions(cls):
        """
        Inputs:
        None.

        Outputs:
        supported_extensions (list of str): list of all extensions (WITH period; e.g., '.edf' or '.pkl') that are accepted by the EEGLoader subclasses we have currently implemented 
        """
        supported_extensions = []
        for Loader in cls._loaders: 
            supported_extensions.extend(Loader.get_supported_extensions())
        return supported_extensions


class EEGThemeParkLoader(EEGLoader):
    """
    Instance of EEGLoader class.
    """
    def load(self, file_path: Path, **kwargs) -> EEGSignal:
        #Implementation of EEGLoader to load files saved from EEG Theme Park

        extension = file_path.suffix #Get the file extension of file_path (including the period) as a string
        if not extension in self.get_supported_extensions():
            #Create a simple dialogue (keeping in mind that )
            pass
        else:
            with open(file_path, "rb") as f:
                eeg_signal_obj = pkl.load(f)
            return (eeg_signal_obj,file_path)
    
    @staticmethod #To allow for class calling without an instance when determining whether or not this loader is appropriate for the given file
    def get_supported_extensions():
        return [".pkl"]

class EDFLoader(EEGLoader):
    global sourcers 
    sourcers = ["detti", "wrlab"]
    
    def load(self, file_path: Path, **kwargs) -> EEGSignal:
        """
        NB: accepts "channel" and "sourcer" from kwargs. channel is the EEG channel we are using, while sourcer is a string that specifies what protocol we should use to gather flags. "unspecified" for sourcer will not prompt any flag retrieval
        
        Inputs:
        - file_path (Path object): full file name as a path object

        Outputs:
        - (eeg_signal_obj, file_path): as a tuple, the EEGSignal object and the original file path
        """
        channel = kwargs.get('channel', None)
        sourcer = kwargs.get("sourcer", None)  # Get cached value
        
        extension = file_path.suffix
        file_name = file_path.stem
        if not extension in self.get_supported_extensions():
            pass
        else:            
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            srate = raw.info['sfreq']
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            channel_names = [raw.ch_names[i] for i in eeg_picks]
            channel, channel_i = self.get_channel(channel_names, kwargs.get('channel'), **kwargs)
            all_data = raw.get_data(picks="eeg") * 1e6  # shape: (n_channels, n_samples)
            
            # Build all_channel_data dict
            all_channel_data = {}
            for i, ch_name in enumerate(channel_names):
                all_channel_data[ch_name] = all_data[i, :]
            
            start_time = raw.first_time
            recording_start = raw.info['meas_date']
            channel_name = channel
            eeg_specs = {
                "name": file_name,
                "channel": channel_name,
                "sourcer":sourcer,
                "srate": srate,
                "data": all_channel_data[channel_name],  # primary channel data
                "all_channel_data": all_channel_data,
                "all_channel_labels": channel_names,
                "start_time": start_time,
                "datetime_collected":recording_start,
                "flags": {},
                "log": ""
            }
            eeg_signal_obj = EEGSignal(**eeg_specs)
            
            for i in range(len(raw.annotations)):
                name = raw.annotations.description[i]
                onset = raw.annotations.onset[i]
                duration = raw.annotations.duration[i]
                
                if duration == 0:
                    times = [onset]
                else:
                    times = [onset, onset + duration]
                
                eeg_signal_obj.add_flag(name, times)
            
            if sourcer is None and len(sourcers)>0: #NB: sourcers are used for files where event data is NOT contained in the EDF annotations and we need special code to load them in. This is most often the case in demo files where event data are stored in strange ways (e.g., .txt files). Note that events/times stored in .xlsx files (so long as they are in the right format) can be loaded using the batch processing mode. If sourcer are left as "unspecified", the EDF will still load correctly (with its annotations stored as flags), there will just be no bespoke importation of any extra events.
                sourcer = text_entry("Where do these files come from? Current options are 'detti' and 'wrlab'.")
                sourcer = [sourcer if sourcer in sourcers else "unspecified"][0]
                eeg_signal_obj.sourcer = sourcer
            else:
                eeg_signal_obj.sourcer = "unspecified"
            
            return (eeg_signal_obj, file_path)
    
    @staticmethod #To allow for class calling without an instance when determining whether or not this loader is appropriate for the given file
    def get_supported_extensions():
        return [".edf"]

class EEGLABLoader(EEGLoader):
    
    def load(self, file_path: Path, **kwargs) -> EEGSignal:
        
        provided_channel = kwargs.get("channel", None)
        
        # Try raw first; if the file contains epochs, fall back to epochs reader
        try:
            raw = mne.io.read_raw_eeglab(str(file_path), preload=True)
            is_epoched = False
        except TypeError:
            epochs = mne.io.read_epochs_eeglab(str(file_path))
            epochs.load_data()
            is_epoched = True
        
        if is_epoched:
            info = epochs.info
            channel_names = epochs.ch_names
            # Concatenate epochs: (n_epochs, n_channels, n_times) -> (n_channels, n_total_samples)
            epoch_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
            concatenated = np.concatenate(epoch_data, axis=-1)  # (n_channels, n_total_samples)
            all_channel_data = {}
            for i, ch_name in enumerate(channel_names):
                all_channel_data[ch_name] = concatenated[i]
        else:
            info = raw.info
            channel_names = raw.ch_names
            all_channel_data = {}
            for i, ch_name in enumerate(channel_names):
                all_channel_data[ch_name] = raw.get_data(picks=[i])[0]
        
        srate = info['sfreq']
        
        # Channel selection (uses the inherited helper)
        selected_channel_name, selected_channel_idx = self.get_channel(
            channel_names, provided_channel=provided_channel, **kwargs
        )
        
        # Extract start time / datetime from MNE info
        meas_date = info.get('meas_date', None)

        if meas_date is not None:
            if hasattr(meas_date, 'timestamp'):  # datetime object
                datetime_collected = meas_date.replace(tzinfo=None) if meas_date.tzinfo else meas_date
            else:
                datetime_collected = dt.datetime(2000, 1, 1, 0, 0, 0)
        else:
            datetime_collected = dt.datetime(2000, 1, 1, 0, 0, 0)
        
        # Extract events/annotations as flags
        flags = {}
        if is_epoched:
            info = epochs.info
            channel_names = epochs.ch_names
            epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            n_epochs, n_channels, n_times_per_epoch = epoch_data.shape
            srate = info['sfreq']

            # epochs.events[:, 0] are sample indices in the original raw recording
            # epochs.tmin is the offset (in seconds) before the event
            tmin_samples = int(round(epochs.tmin * srate))
            epoch_starts = epochs.events[:, 0] + tmin_samples  # absolute sample index of each epoch's first sample

            # Total length: from first epoch start to end of last epoch
            total_samples = (epoch_starts[-1] + n_times_per_epoch) - epoch_starts[0]

            # Initialize with NaN and fill in epoch data at correct positions
            full_data = np.full((n_channels, total_samples), np.nan)
            offset = epoch_starts[0]  # baseline so first epoch starts at index 0
            for i in range(n_epochs):
                start_idx = epoch_starts[i] - offset
                full_data[:, start_idx:start_idx + n_times_per_epoch] = epoch_data[i]

            all_channel_data = {}
            for i, ch_name in enumerate(channel_names):
                all_channel_data[ch_name] = full_data[i] * 1e6

            # Invert event_id mapping: {int_code: "label_string"}
            id_to_label = {v: k for k, v in epochs.event_id.items()}

            # Flags: mark each event at its true position and each epoch window
            flags = {}
            
            #Code to shade epochs; omitted, as it was cluttering up the flags view
            # for i in range(n_epochs):
            #     int_code = epochs.events[i, 2]
            #     label = id_to_label.get(int_code, str(int_code))

            #     # Point event at the actual event position
            #     event_sample = epochs.events[i, 0] - offset
            #     flags[f"{label}_epoch{i}"] = [event_sample / srate]

            #     # Epoch window as a shaded range
            #     epoch_start_sec = (epoch_starts[i] - offset) / srate
            #     epoch_end_sec = epoch_start_sec + (n_times_per_epoch / srate)
            #     flags[f"{label}_epoch{i}_window"] = [epoch_start_sec, epoch_end_sec, True]

            annotations = None
        else:
            annotations = raw.annotations
        if annotations is not None and len(annotations) > 0:
            for ann in annotations:
                label = ann['description']
                onset_sec = ann['onset']
                duration = ann.get('duration', 0.0)
                if duration > 0:
                    flags[label] = [onset_sec, onset_sec + duration, False]
                else:
                    flags[label] = [onset_sec]
        
        # Build signal_specs
        signal_specs = {
            "name": file_path.stem,
            "channel": selected_channel_name,
            "srate": srate,
            "data": all_channel_data[selected_channel_name],
            "start_time": 0,
            "datetime_collected": datetime_collected,
            "flags": flags,
            "log": [],
            "all_channel_data": all_channel_data,
            "all_channel_labels": channel_names,
        }
        
        eeg_signal_obj = EEGSignal(**signal_specs)
        
        return (eeg_signal_obj, file_path)
    
    @staticmethod
    def get_supported_extensions():
        return [".set"]

