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
        
        # Normalize provided channel
        if provided_channel is not None:
            provided_channel = provided_channel.upper()
        
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
    
    def add_detti_flags(self, eeg_signal_obj):
        
        search_name = eeg_signal_obj.name
        
        # Find the .txt file
        project_root = Path(__file__).parent.parent
        txt_files = list(project_root.rglob(f"{search_name}.txt"))
        if not txt_files:
            return
        if len(txt_files) > 1:
            raise ValueError(f"Multiple .txt files found for {search_name}: {txt_files}")
        txt_file = txt_files[0]
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Helper to parse time strings
        def parse_time_str(time_str):
            time_str = time_str.strip().replace('.', ':')
            parts = time_str.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid time format: {time_str}")
            hours, minutes, seconds = map(int, parts)
            return dt.time(hours, minutes, seconds)
        
        # Extract registration start time from .txt (ground truth)
        reg_start_match = re.search(r'Registration start time:\s*([0-9:.]+)', content)
        if not reg_start_match:
            raise ValueError(f"Could not find registration start time in {txt_file}")
        
        reg_start_time = parse_time_str(reg_start_match.group(1))
        
        # Verify signal hasn't been cropped
        if eeg_signal_obj.start_time != 0:
            raise ValueError(
                f"Signal has been cropped (start_time = {eeg_signal_obj.start_time}). "
                f"Cannot reliably align with .txt annotations.")
        
        # Override datetime_collected to match .txt ground truth (use arbitrary date, timezone-naive)
        eeg_signal_obj.datetime_collected = dt.datetime.combine(
            dt.date(2000, 1, 1),  # Arbitrary date
            reg_start_time
        )
        
        # Verify recording is <24 hours (otherwise hours repeat and are ambiguous)
        recording_duration_hours = len(eeg_signal_obj.data) / eeg_signal_obj.srate / 3600
        if recording_duration_hours >= 24:
            raise ValueError(
                f"Recording duration ({recording_duration_hours:.2f} hours) >= 24 hours. Cannot unambiguously determine seizure dates from time-only annotations.")
        
        # Find all seizure blocks
        seizure_pattern = r'Seizure n \d+.*?(?=Seizure n \d+|$)'
        seizures = re.findall(seizure_pattern, content, re.DOTALL)
        
        if not seizures:
            return  # No seizures found
        
        recording_start_datetime = eeg_signal_obj.datetime_collected
        # Make timezone-naive since .txt annotations don't include timezone info
        if recording_start_datetime.tzinfo is not None:
            recording_start_datetime = recording_start_datetime.replace(tzinfo=None)
        recording_start_date = recording_start_datetime.date()
        
        for seizure_text in seizures:
            # Extract start time (handles both format variations)
            start_match = re.search(
                r'(?:Seizure start time|Start time):\s*([0-9:.]+)', 
                seizure_text
            )
            end_match = re.search(
                r'(?:Seizure end time|End time):\s*([0-9:.]+)', 
                seizure_text
            )
            
            if not start_match or not end_match:
                raise ValueError(f"Could not extract seizure times from: {seizure_text[:150]}")
            
            seizure_start_time = parse_time_str(start_match.group(1))
            seizure_end_time = parse_time_str(end_match.group(1))
            
            # Determine dates: if time < recording start time, must be next day
            def time_to_datetime(time_obj):
                if time_obj >= recording_start_datetime.time():
                    return dt.datetime.combine(recording_start_date, time_obj)
                else:
                    return dt.datetime.combine(
                        recording_start_date + dt.timedelta(days=1), 
                        time_obj
                    )
            
            seizure_start_datetime = time_to_datetime(seizure_start_time)
            seizure_end_datetime = time_to_datetime(seizure_end_time)
            
            # If end <= start, end must be next day
            if seizure_end_datetime <= seizure_start_datetime:
                seizure_end_datetime = dt.datetime.combine(
                    seizure_start_datetime.date() + dt.timedelta(days=1),
                    seizure_end_time
                )
            
            # Validate seizure is within recording bounds
            recording_end_datetime = (
                recording_start_datetime + 
                dt.timedelta(seconds=len(eeg_signal_obj.data) / eeg_signal_obj.srate)
            )
            
            if seizure_start_datetime < recording_start_datetime:
                raise ValueError(
                    f"Seizure start {seizure_start_datetime} before recording start {recording_start_datetime}"
                )
            
            if seizure_end_datetime > recording_end_datetime:
                raise ValueError(
                    f"Seizure end {seizure_end_datetime} after recording end {recording_end_datetime}"
                )
            
            # Add the flag with shading
            flag_arr = [seizure_start_datetime, seizure_end_datetime]
            eeg_signal_obj.add_flag("Seizure", flag_arr, shade=True)
    
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
        if not channel is None:
            channel = channel.upper()
        
        extension = file_path.suffix
        file_name = file_path.stem
        if not extension in self.get_supported_extensions():
            pass
        else:            
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            srate = raw.info['sfreq']
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            channel_names = [raw.ch_names[i].upper() for i in eeg_picks]
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
            
            if sourcer is None: #NB: sourcers are used for files where event data is NOT contained in the EDF annotations and we need special code to load them in. This is most often the case in demo files where event data are stored in strange ways (e.g., .txt files). Note that events/times stored in .xlsx files (so long as they are in the right format) can be loaded using the batch processing mode. If sourcer are left as "unspecified", the EDF will still load correctly (with its annotations stored as flags), there will just be no bespoke importation of any extra events.
                sourcer = text_entry("Where do these files come from? Current options are 'detti' and 'wrlab'.")
                sourcer = [sourcer if sourcer in sourcers else "unspecified"][0]
                eeg_signal_obj.sourcer = sourcer
             
            if sourcer == "detti":
                self.add_detti_flags(eeg_signal_obj)
            
            return (eeg_signal_obj, file_path)
    
    @staticmethod #To allow for class calling without an instance when determining whether or not this loader is appropriate for the given file
    def get_supported_extensions():
        return [".edf"]

