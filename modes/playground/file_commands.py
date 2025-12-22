"""
This module handles the creation, editing and storing of different signals that we create. Expressly, it DOES NOT yet handle the editing and management of Excel files that store data surrounding signal start and end times.
"""
from pathlib import Path as path
import numpy as np
import pickle as pkl
from tkinter import filedialog
from eeg_theme_park.utils import gui_utilities, eeg_signal
from pathlib import Path
from eeg_theme_park.utils.loaders import AllLoaders

def build_signal_object(signal_specs: dict = None, flags: dict = {}) -> object:
    """
    Function that returns an EEG signal with the specifications contained in specs. NB: does NOT save the signal or initialize the signal cache (this is done in the save_signal() method).

    Inputs:
    - signal_specs (dict): a dictionary containing all the data needed to initialize a signal; should contain keys "name", "amp", "freq", "phase", "srate", "time_length", and "start_time"
    - time_specs (dict): a dictionary containing the time markers of interest in the form marker_name : arr, where marker_name is a string and arr is an array of one or two values in seconds (one if an event is a one-off, two if an event has a start and finish)
    
    Outputs:
    - EEGSignal object (the signal based on signal specs)
    """
    #Step 0: Quality control on inputs
    required_keys = ["name", "amp", "freq", "phase", "srate", "time_length"]
    if signal_specs is not None and all(k in signal_specs and signal_specs[k] is not None for k in required_keys): #If signal_specs exists and has all fields filled out, no need to prompt the user for information
        pass
    else:
        signal_specs = gui_utilities.get_specs(signal_specs=signal_specs) #Allows the user to input their desired specs if signal_specs does not contain all the specs needed
    if signal_specs is None: #Case where the user closes the signal_specs window
        return

    #Step 1: This code will add to our signal_data excel file; currently, this has not been implemented. In fact, it may not be necessary to, as the flags variable is passed to sig_to_save and, in theory, should contain data surrounding the various time points.
    if not "start_time" in signal_specs.keys():
        time_start = 0
        time_end = time_start+signal_specs["time_length"] #In seconds
        flags = {}
    else: #Adds time flags
        time_start=signal_specs.get("start_time",0) #Returns "start_time" value from time specs; if does not exist, returns 0
        time_end = time_start+signal_specs["time_length"]
        flags = {key:value for key, value in flags.items() if key not in ["name","srate","time_len","freq","amp","phase","start_time"]} #flags' entries will be in the form str : arr, where str is the event name, and and arr is the start (or start and stop) time of the event

    #Step 2: Build, pickle, and save the signal
    a = signal_specs["amp"]
    f = signal_specs["freq"] #In Hz
    omega = 2*np.pi*f #angular frequency; multiply by 2pi to convert from cycles/sec to rad/sec (2pi rad in a circle, or cycle)
    phi = signal_specs["phase"]
    num_samples = (time_end-time_start)*signal_specs["srate"]
    times = np.linspace(time_start, time_end, int(num_samples)) #Creates our times
    complex_signal = a*np.exp(1j * (omega*times + phi)) #We now have a complex signal to our specifications; cos is the real part, sin is the imaginary part
    signal = np.imag(complex_signal)
    del complex_signal
    log_text = f"Signal {signal_specs['name']} initialized with amplitude {a} uV, frequency {f} Hz, phase {phi} rad, length {(num_samples/signal_specs['srate']):.2f} sec, and starting at {time_start} sec."
    signal_data = signal_specs.copy()  # Start with all input specs
    signal_data.update({  # Override/add the generated fields
        "data": signal,
        "start_time": time_start,
        "flags": flags,
        "log": log_text
    })
    
    return eeg_signal.EEGSignal(**signal_data)

def save_signal(eeg_signal_obj: object, file_path : path):
    """
    Saves an EEGSignal object to the disk. NB: also creates a signal cache directory if needed for signal.
    """
    #NB: the user may save signals to wherever they desire; when building in analytic functionality, therefore, it will be incredibly important that the user includes all files to analyze in a single directory.
    with open(file_path, "wb") as f:
        pkl.dump(eeg_signal_obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def load_signal(choose: bool = False, file_path: path = None, file_name_bool = None, **kwargs) -> tuple[object, path]:
    """
    This method allows the user to choose a pickled signal file to load, then reconstructs and reutrns the signal.

    Inputs:
    - parent (path): parent path where the 
    - choose (bool): if False, the code will proceed to open the file at file_path
    - file_name_bool (bool): if True, in the case the a stored EEGSignal object's name doesn't match its file name, we replace the stored name with the file name (minus the extension)

    Outputs:
    - (eeg_signal_obj, file_path): returns EEGSignal object and file_path reflecting where it as saved, both as a tuple
    """
    if choose or file_path is None: #Allow user to specify file_path
        # Get supported extensions and format for filetypes
        supported_extensions = AllLoaders.get_supported_extensions()
        filetypes = [(f"{ext.upper()} files", f"*{ext}") for ext in supported_extensions]
        filetypes.append(("All files", "*.*"))
        
        file_path = Path(filedialog.askopenfilename(
            parent=file_path,
            defaultextension=supported_extensions[0] if supported_extensions else ".pkl",
            filetypes=filetypes,
            initialdir=path.cwd(),
            title="Load EEG Signal"
        ))
    
    if str(file_path) in ['',".","None"]:
        if not(gui_utilities.yes_no("It seems you didn't choose a file. Cancel file loading?")):
            return load_signal(choose=True)
        else:
            return (None, None)
    
    #Open and return file
    all_suffixes = AllLoaders.get_supported_extensions()
    if file_path.suffix not in all_suffixes:
        cont_bool = gui_utilities.yes_no(f"The file format you chose ({file_path.suffix}) is not currently supported for loading. Would you like to load a different file (Yes) or exit loading (No)?")
        if not cont_bool:
            return (None,None)
        else:
            return load_signal(choose=True)
    #Find appropriate loader
    for loader in AllLoaders._loaders: #loader is a class, not an instance
        if file_path.suffix in loader.get_supported_extensions():
            loader_inst = loader()
            eeg_signal_obj = loader_inst.load(file_path, **kwargs)[0] #First in tuple is EEGSignal instance, second in tuple is file_path (which we already have)
            my_ext = file_path.suffix
    file_name = path(file_path).stem
    object_name = eeg_signal_obj.name
    #Handle cases where file name does not match saved EEG metadata name
    if not file_name == object_name:
        if file_name_bool is None:
            file_name_bool = gui_utilities.yes_no(f"The name of the file ({file_name}) does not match the name stored in the saved EEG signal ({object_name}). Would you like to proceed with the file name? (Yes will proceed with the file name {file_name}, No will proceed with the stored signal name {object_name}).")
        if file_name_bool:
            eeg_signal_obj.name = file_name
            gui_utilities.simple_dialogue("The signal's name has been updated. It can be renamed via the main menu.")
        else: #In this case, we retain the signal's name but have to rename the file
            new_file_name = eeg_signal_obj.name #object_name is just eeg_signal_obj.name
            new_file_path = file_path.parent / f"{new_file_name}{my_ext}"
            # Check if new_file_path exists and prompt for a different name if needed
            while new_file_path.exists():
                new_file_name = gui_utilities.text_entry(f"{new_file_name}{my_ext} already exists in this folder. Please choose a different name for the file.")
                new_file_path = file_path.parent / f"{new_file_name}{my_ext}"
            # Update signal name, delete old file, and save with new name
            eeg_signal_obj.name = new_file_name
            file_path.unlink()  # Delete the old file
            save_signal(eeg_signal_obj, new_file_path)
            return (eeg_signal_obj, new_file_path)
    return (eeg_signal_obj, file_path)

if __name__ == "__main__":
    signal_specs={"name" : "signal_1", "amp":50, "freq":30, "phase" : 0, "srate":250, "time_length":3600}
    build_signal_object(signal_specs=signal_specs)