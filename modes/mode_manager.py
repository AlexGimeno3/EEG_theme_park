"""
A note on architecture: the root window (a tk.Tk instance) is the single OS window that the application will be displayed in. There will be different frames (containers of visual information; these are ttk.Frame instances) that we will be swapping in and out of the root window; each mode, therefore, will be contained in a ttk.Frame instance. \
"""


import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
from pathlib import Path
import pickle as pkl
import os
from eeg_theme_park.utils import gui_utilities, loaders

class Mode(ABC, ttk.Frame):
    """
    Abstract class that each mode inherits and then implements.

    Inputs:
    - 
    """
    name = None #Name for this mode; can be "playground", ...

    def __init__(self, parent, mode_manager):
        """
        Mode initialization.

        Inputs:
        - parent (any tkinter widget): the widget we will be using to hold/encapsulate this mode (see architecture note above); therefore, this parent is one above the current Mode's Frame in the widget heirarchy
        - mode_manager (ModeManager instance): reference to the ModeManager we are using to manage the relationship between our Mode subclasses and the root window.
        """ 
        super().__init__(parent) #Calls the init method of the next class in the MRO; here, that is ABC, which has no __init__() method implemented, so we move onto ttk.Frame's __init__ method; this is why we need to pass parent, as any time a ttk.Frame is initialized, it needs a parent passed to know where it lives.
        self.mode_manager = mode_manager
        self.supported_extensions = loaders.AllLoaders.get_supported_extensions()
        self.parent = parent
        self.path = None #The path where the Mode is saved to. Initially starts as None, and is updated in the ModeManager class on saving.
        self.unsaved_changes_present = False
        self._is_initialized = False #Set to True after we call the self.show() method, which initializes the mode prior to the first time it is displayed

    def __init_subclass__(cls, **kwargs):
        #Ensure that, for every Mode subclass that is defined throughout the program, the subclass is immediately registered in AllModes and therefore accessible via AllModes._modes
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise TypeError(f"You must instantiate a 'name' class attribute when instantiating a Mode subclass.")
        if cls.name in [mode.name for mode in AllModes._modes]:
            raise TypeError(f"Modes cannot have the same name; however, Python detected two modes with the name {cls.name}. Please check the different instantaitions of the Mode class to be sure you haven't added a mode duplicating the name {cls.name}.")
        AllModes.add_to_modes(cls)

    def show(self):
        """
        Called when a mode is activated; can be overriden, but should encapsulate the initialization process pretty well.
        """
        if not self._is_initialized:
            self.initialize_ui() #Abstract method to initialize
            self._is_initialized = True
        self.pack(fill=tk.BOTH, expand = True) #Uses the pack pethod inherited from ttk.Frame
        self.update_display() #Abstract method to update

    def hide(self):
        """
        Called when leaving a mode.
        """
        self.pack_forget() #Eliminates self (the mode's Frame from the root window [pack is used since self.pack is used in show()])

    
    @abstractmethod
    def initialize_ui(self):
        """
        Initialize the UI elements for this mode the first time it is rendered. Called just once.
        """
        pass

    @abstractmethod
    def update_display(self):
        """
        Updates the mode's Frame.
        """
        pass

class AllModes:
    """
    Stores/manages all mode subclasses.
    """
    _modes = []

    @classmethod
    def add_to_modes(cls, Mode):
        """
        Adds a mode subclass to _modes

        Inputs:
        - Mode (Mode class): subclass of Mode to add to _modes

        Outputs:
        None.
        """
        cls._modes.append(Mode)
    
    @classmethod
    def get_mode_names(cls):
        return [mode.name for mode in cls._modes]
    
    @classmethod
    def str_to_mode(cls, name_str):
        """Gets the name of a mode and returns the Mode instance"""
        #Return the mode in cls._modes where mode.name == name_str
        for mode in cls._modes:
            if mode.name == name_str:
                return mode
        return None


class ModeManager:
    """
    Stores/manages multiple EEG Theme Park mode Frames and handles switching between them.
    """
    def __init__(self, root):
        """
        Initialize the mode manager.

        Inputs:
        - root (tk.Tk instance): the main Tk root window into which all the other modes will be imported
        """
        self.root = root
        self.modes = [mode(root, self) for mode in AllModes._modes] #Instantiates all modes from the get-go
        self.current_mode = [mode for mode in self.modes if mode.name == "playground"][0] #Initialize first mode

        #Create menubar
        self.menubar = tk.Menu(root)
        root.config(menu=self.menubar)
        
        #File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Session", menu=self.file_menu)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit)
        
        #Create mode selector frame at the top of the root window
        self.selector_frame = ttk.Frame(root)
        self.selector_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Label(self.selector_frame, text="Mode:").pack(side=tk.LEFT, padx=5)

        self.selected_mode = tk.StringVar() #Bidirectional variable to store the user's dropdown selection for the mode being used
        self.mode_dropdown = ttk.Combobox(self.selector_frame,textvariable=self.selected_mode, state = "readonly", width = 30)
        self.mode_dropdown['values'] = [mode.name for mode in self.modes] #Mode names to display
        self.mode_dropdown.bind('<<ComboboxSelected>>', self._switch_to_mode) #Triggers when mode is selected
        self.mode_dropdown.pack(side=tk.LEFT, padx=5)

        self.mode_dropdown.set(self.current_mode.name)
        self.current_mode.show()


        #TODO: Add autosave


    def __getstate__(self):
        return {
            "modes_state": [mode.__getstate__() for mode in self.modes],
            "current_mode_name": self.current_mode.name
        }
    
    def __setstate__(self, state):
        self._saved_state = state

    def attach_to_root(self, root):
        self.__init__(root) #Rebuilds GUI structure
        for mode, saved_state in zip(self.modes, self._saved_state["modes_state"]):
            mode._restore_mode_data(saved_state["mode_data"])
    
    def _switch_to_mode(self, event = None):
        """
        Handles mode selection from the Combobox (drop-down)
        """
        self.current_mode.hide()
        mode_name = self.selected_mode.get() #This returns a string with the Mode's name
        self.current_mode = [mode for mode in self.modes if mode.name == mode_name][0] #Switch mode
        self.current_mode.show()

    def exit(self):
        self.root.quit



                



    
