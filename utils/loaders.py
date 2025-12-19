"""
Interface-based module containing various loaders for EEG files of different formats.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from eeg_theme_park.utils.eeg_signal import EEGSignal
import pickle as pkl
import mne
from mne._fiff import utils
from eeg_theme_park.utils.gui_utilities import choose_channel
from eeg_theme_park.utils.misc_utils import get_start_time_from_excel


# Create the permissive read function
def _read_str_permissive(fid, count):
    """Read a string from a file with permissive encoding."""
    bytestr = fid.read(count).split(b'\x00', 1)[0]
    
    for encoding in ['ascii', 'latin-1', 'utf-8', 'cp1252', 'iso-8859-1']:
        try:
            return str(bytestr.decode(encoding))
        except (UnicodeDecodeError, AttributeError):
            continue
    
    return str(bytestr.decode('latin-1', errors='ignore'))

# Apply the patch globally - this affects all subsequent MNE operations
utils.read_str = _read_str_permissive

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
    

class BDFLoader(EEGLoader):
    def load(self, file_path: Path, **kwargs) -> EEGSignal:
        """
        NB: accepts "channel" from kwargs
        """
        channel = kwargs.get('channel', None)
        if not channel is None:
            channel = channel.upper()
        
        extension = file_path.suffix #Get the file extension of file_path (including the period) as a string
        if not extension in self.get_supported_extensions():
            pass
        else:
            raw = mne.io.read_raw_bdf(file_path, preload=False, verbose=False)
            srate = raw.info['sfreq']
            channel_names = [name.upper() for name in raw.ch_names]
            print(channel_names)
            if (channel is None) or (not channel in channel_names):
                channel, channel_i = choose_channel(channel_names)
            data = raw.get_data(picks="eeg")
            data = data[channel_i, :]
            start_time = raw.first_time
            eeg_specs = {
                "srate":srate,
                "data":data,
                "start_time":start_time,
                "flags":{},
                "log":""
            }
            eeg_signal_obj = EEGSignal(**eeg_specs)
            
            for i in range(len(raw.annotations)):
                name = raw.annotations.description[i]
                onset = raw.annotations.onset[i]
                duration = raw.annotations.duration[i]
                
                if duration == 0:
                    # One-off marker
                    times = [onset]
                else:
                    # Duration event
                    times = [onset, onset + duration]
                
                eeg_signal_obj.add_flag(name, times)

            return (eeg_signal_obj,file_path)
    
    @staticmethod #To allow for class calling without an instance when determining whether or not this loader is appropriate for the given file
    def get_supported_extensions():
        return [".bdf"]
    

class DettiEDFLoader(EEGLoader):
    def add_flags(eeg_signal_obj):
        """
        Helper function that finds and extracts seizure (flag) data from auxiliary .txt files to be added to the signal later on.

        Inputs:
        eeg_signal_obj (EEGSignal subclass): Detti EDF eeg_signal_object we will be finding flags for.
        """

    
    def load(self, file_path: Path, **kwargs) -> EEGSignal:
        """
        NB: accepts "channel" from kwargs
        """
        channel = kwargs.get('channel', None)
        if not channel is None:
            channel = channel.upper()
        
        extension = file_path.suffix #Get the file extension of file_path (including the period) as a string
        file_name = file_path.stem
        if not extension in self.get_supported_extensions():
            pass
        else:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            srate = raw.info['sfreq']
            channel_names = [name.upper() for name in raw.ch_names]
            print(channel_names)
            if (channel is None) or (not channel in channel_names):
                channel, channel_i = choose_channel(channel_names)
            data = raw.get_data(picks="eeg")
            data = data[channel_i, :]
            start_time = raw.first_time
            recording_start = 
            eeg_specs = {
                "name":file_name
                "srate":srate,
                "data":data,
                "start_time":start_time,
                "real_world_start_time":recording_start
                "flags":{},
                "log":""
            }
            eeg_signal_obj = EEGSignal(**eeg_specs)
            
            for i in range(len(raw.annotations)):
                name = raw.annotations.description[i]
                onset = raw.annotations.onset[i]
                duration = raw.annotations.duration[i]
                
                if duration == 0:
                    # One-off marker
                    times = [onset]
                else:
                    # Duration event
                    times = [onset, onset + duration]
                
                eeg_signal_obj.add_flag(name, times)

            self.add_flags(eeg_signal_obj)
            return (eeg_signal_obj,file_path)
    
    @staticmethod #To allow for class calling without an instance when determining whether or not this loader is appropriate for the given file
    def get_supported_extensions():
        return [".detti_edf"]

