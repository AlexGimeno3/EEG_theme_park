"""
This module contains code to support GUI features of the application; these have been sequestered here because they can be busy in the main script files
"""
import tkinter as tk
from tkinter import ttk
from eeg_theme_park.modes.playground import file_commands
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pathlib import Path
import copy
import numpy as np
from eeg_theme_park.modes.mode_manager import Mode, ModeManager
from eeg_theme_park.utils import eeg_analyzers, signal_functions, gui_utilities
from eeg_theme_park.utils.gui_utilities import choose_channel

class PlaygroundMode(Mode):
    """
    Playground mode for interactive signal generation and analysis.
    """
    name = "playground"

    @property
    def signal_path(self):
        if self.signal_dir is None:
            self.signal_dir = Path(gui_utilities.choose_dir())
        return Path(self.signal_dir)/f"{self.current_signal.name}.pkl"
    
    def __init__(self, parent, mode_manager):
        super().__init__(parent, mode_manager)

        #Initialize state (i.e., EEG signal) variables
        self.current_signal = None #This is an EEGSignal object
        self.display_time_lims = [] #Start and stop times when displaying data in SECONDS
        self.analyze_time_lims = [] #Start and stop times during analysis in SECONDS (will be useful for programatic analysis during only certain periods of the signal, since some metrics will be things like slope or mean)
        self.signal_dir = None #This will store the signal's current directory; the full path (with file name and extension) is handled by the setter signal_path below
        self.min_clean_time = None #Minimum length of consecutive clean data needed to allow for processing (EEGFunctions or EEGAnalyzers)
        self.has_unsaved_changes = False #True when the signal has been changed but user hasn't saved
        self.view_timeseries = [] #List of TimeSeries to view

        #Initialize GUI variables (these will be created in initialize_ui())
        self.canvas = None
        self.toolbar = None
        self.plot_frame = None 

    def initialize_ui(self):
        """
        Initializes UI elements for playground mode; called just once (the first time the mode is shown).
        """
        self._create_menubar() #Creates menubar for Frame if we do not have it already
        self.plot_frame = ttk.Frame(self) #Build the frame for matplotlib
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) #Format matplotlib

    def _create_menubar(self): 
        """
        Intended for one-time use while initializing GUI window.
        Will add more menubars and commands as time goes on
        """
        root = self.mode_manager.root

        if not hasattr(self,"_menubar"): #Stores current menubar to avoid rebuilding when switching between modes
            self._menubar = tk.Menu(root)
            #Build file menu
            file_menu = tk.Menu(self._menubar, tearoff=0)
            self._menubar.add_cascade(label = "File", menu=file_menu)
            file_menu.add_command(label = "Build EEG", command =self.build_signal_cmd)
            file_menu.add_command(label = "Load EEG", command =self.load_signal_cmd)
            file_menu.add_command(label = "Save", command=self.save_signal_cmd)
            file_menu.add_command(label = "Save as", command=lambda: self.rename_cmd(rename_saved=False))
            file_menu.add_command(label = "Rename", command =self.rename_cmd)

            #Signal menu
            signal_menu = tk.Menu(self._menubar, tearoff=0)
            self._menubar.add_cascade(label="Signal", menu=signal_menu)
            signal_menu.add_command(label="Switch channel", command=self.switch_channel_cmd)
            signal_menu.add_command(label="Remove channels", command=self.remove_channels_cmd)
            signal_menu.add_command(label="Alter signal", command=self.alter_signal_cmd)
            signal_menu.add_command(label="Extract feature", command=self.analyze_signal_cmd)
            
            #View menu
            view_menu = tk.Menu(self._menubar, tearoff=0)
            self._menubar.add_cascade(label="View", menu=view_menu)
            view_menu.add_command(label="View time series", command=self.view_ts_cmd)
            view_menu.add_command(label="Zoom time series", command=self.zoom_cmd)
            view_menu.add_command(label="View flags", command=self.view_flags_cmd)
            view_menu.add_command(label="Select flags", command=self.select_flags_cmd)
            view_menu.add_command(label="View signal data", command=self.view_signal_info_cmd)

    def show(self):
        self._create_menubar()
        self.mode_manager.root.config(menu=self._menubar) #Changes root window's menubar
        super().show() #Uses the initialize_ui() method above

    def hide(self):
        super().hide()

    def update_display(self, time_series=[]):
        if self.current_signal is not None:
            self._update_display(time_series)

    #Utility functions ------------------
    def remove_channels_cmd(self):
        """Allow the user to permanently remove channels from the current signal."""
        if self.current_signal is None:
            gui_utilities.simple_dialogue("No signal is loaded. Please load or build a signal first.")
            return
        available_channels = self.current_signal.all_channel_labels
        if len(available_channels) <= 1:
            gui_utilities.simple_dialogue(
                f"This signal only has one channel ({available_channels[0]}), so there are no channels to remove.")
            return

        # Build dialogue window
        dialogue = tk.Toplevel(self)
        dialogue.title("Remove Channels")
        dialogue.geometry("400x500")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        main_label = ttk.Label(dialogue, text="Select channels to permanently remove:")
        main_label.pack(pady=10)

        # Create frame for checkboxes with scrollbar
        checkbox_frame_outer = ttk.Frame(dialogue)
        checkbox_frame_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(checkbox_frame_outer)
        scrollbar = ttk.Scrollbar(checkbox_frame_outer, orient="vertical", command=canvas.yview)
        checkbox_frame = ttk.Frame(canvas)

        checkbox_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        checkbox_vars = {}
        for idx, name in enumerate(available_channels):
            var = tk.BooleanVar(value=False)
            checkbox_vars[name] = var
            cb = ttk.Checkbutton(checkbox_frame, text=name, variable=var)
            cb.grid(row=idx, column=0, sticky="w", padx=5, pady=2)

        def submit():
            channels_to_remove = [name for name, var in checkbox_vars.items() if var.get()]
            if not channels_to_remove:
                gui_utilities.simple_dialogue("No channels were selected for removal.")
                return
            # Ensure at least one channel remains
            remaining = [ch for ch in available_channels if ch not in channels_to_remove]
            if len(remaining) == 0:
                gui_utilities.simple_dialogue("You cannot remove all channels. At least one channel must remain.")
                return
            # # Confirm with user
            # confirm = gui_utilities.yes_no(
            #     f"Are you sure you want to permanently remove the following channel(s)?\n\n"
            #     f"{', '.join(channels_to_remove)}\n\n"
            #     f"This action cannot be undone."
            # )
            # if not confirm:
            #     return
            dialogue.destroy()

            # If current channel is being removed, switch first
            if self.current_signal.current_channel in channels_to_remove:
                self.current_signal.switch_channel(remaining[0])

            # Remove from all_channel_data and all_channel_labels
            for ch in channels_to_remove:
                del self.current_signal.all_channel_data[ch]
                self.current_signal.all_channel_labels.remove(ch)

            # Clean up TimeSeries objects that have per-channel data
            for ts in self.current_signal.time_series:
                if ts.channel_data is not None:
                    for ch in channels_to_remove:
                        ts.channel_data.pop(ch, None)
                    # If the TimeSeries's primary_channel was removed, reassign it
                    if ts.primary_channel in channels_to_remove:
                        if ts.channel_data:
                            ts.primary_channel = next(iter(ts.channel_data))
                        else:
                            ts.primary_channel = None

            self.current_signal.has_unsaved_changes = True
            self.current_signal.log_text(f"Removed channel(s): {', '.join(channels_to_remove)}.")
            self.update_display(time_series=self.view_timeseries)

        # Select all / deselect all buttons
        button_frame = ttk.Frame(dialogue)
        button_frame.pack(pady=(0, 5))

        def select_all():
            for var in checkbox_vars.values():
                var.set(True)

        def deselect_all():
            for var in checkbox_vars.values():
                var.set(False)

        ttk.Button(button_frame, text="Select all", command=select_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Deselect all", command=deselect_all).pack(side="left", padx=5)

        submit_button = ttk.Button(dialogue, text="Remove Selected", command=submit)
        submit_button.pack(pady=20)

        dialogue.wait_window()
    
    def switch_channel_cmd(self):
        """Allow the user to switch the active channel for the current signal."""
        if self.current_signal is None:
            gui_utilities.simple_dialogue("No signal is loaded. Please load or build a signal first.")
            return
        available_channels = self.current_signal.all_channel_labels
        if len(available_channels) <= 1:
            gui_utilities.simple_dialogue(
                f"This signal only has one channel ({available_channels[0]}), so there is nothing to switch to.")
            return

        selection = choose_channel(available_channels)
        if selection is None:
            return
        new_channel_name, _ = selection
        if new_channel_name == self.current_signal.current_channel:
            gui_utilities.simple_dialogue(f"'{new_channel_name}' is already the active channel.")
            return
        old_channel = self.current_signal.current_channel
        self.current_signal.switch_channel(new_channel_name)
        self.current_signal.log_text(f"Active channel switched from {old_channel} to {new_channel_name}.")
        self.update_display()
    
    def view_signal_info_cmd(self):
        """Display signal information in a formatted dialog."""
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You need to build or import a signal before you can view its information.")
            return
        
        # Format datetime if it exists
        datetime_str = "Not recorded"
        if hasattr(self.current_signal, 'datetime_collected') and self.current_signal.datetime_collected:
            try:
                datetime_str = self.current_signal.datetime_collected.strftime("%Y-%m-%d %H:%M:%S")
            except (AttributeError, ValueError):
                datetime_str = str(self.current_signal.datetime_collected)
        
        # Build the information message
        info_lines = [
            "Signal Information:",
            "",
            f"Name: {self.current_signal.name}",
            f"Start Time: {self.current_signal.start_time:.3f} ms",
            f"Datetime Collected: {datetime_str}",
            f"Sampling Rate: {self.current_signal.srate} Hz",
            "",
            "Log:",
            "─" * 50
        ]
        
        # Add log entries
        if hasattr(self.current_signal, 'log') and self.current_signal.log:
            info_lines.extend(self.current_signal.log.strip().split('\n'))
        else:
            info_lines.append("No log entries.")
        
        message = "\n".join(info_lines)

        #Scrollable dialogue
        dialogue = tk.Toplevel(self)
        dialogue.title("Signal information")
        dialogue.geometry("600x500")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        text_frame = ttk.Frame(dialogue)
        text_frame.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        scrollbar.pack(side="right",fill="y")

        text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set)
        text_widget.insert("1.0",message)
        text_widget.config(state="disabled")
        text_widget.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=text_widget.yview)

        ttk.Button(dialogue, text="Close", command=dialogue.destroy).pack(pady=10)
    
    def view_flags_cmd(self):
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You need to build or import a signal before you can make changes to it.")
            return
        # Format flags dictionary nicely
        if not self.current_signal.flags:
            gui_utilities.simple_dialogue("No flags have been set for this signal.")
            return
        
        # Build formatted message
        message_lines = ["Signal Flags:\n"]
        for flag_name, flag_values in self.current_signal.flags.items():
            times_sec = [t for t in flag_values[:2]]
            
            if len(flag_values) == 1:
                # Single time point
                message_lines.append(f"  • {flag_name}: {times_sec[0]:.3f} s")
            elif len(flag_values) == 3:
                # Time range with shading option
                shade_text = " (shaded)" if flag_values[2] else ""
                message_lines.append(f"  • {flag_name}: {times_sec[0]:.3f} s - {times_sec[1]:.3f} s{shade_text}")
        
        message = "\n".join(message_lines)

        #Scrollable dialogue
        dialogue = tk.Toplevel(self)
        dialogue.title("View Flags")
        dialogue.geometry("500x400")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        text_frame = ttk.Frame(dialogue)
        text_frame.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical")
        scrollbar.pack(side="right",fill="y")

        text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set)
        text_widget.insert("1.0",message)
        text_widget.config(state="disabled")
        text_widget.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=text_widget.yview)

        ttk.Button(dialogue, text="Close", command=dialogue.destroy).pack(pady=10)
    
    def select_flags_cmd(self):
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You need to build or import a signal before you can select flags.")
            return
        if not self.current_signal.flags:
            gui_utilities.simple_dialogue("No flags have been set for this signal.")
            return
        if not hasattr(self.current_signal, "_flag_visibility"):
            self.current_signal._flag_visibility = {}

        flag_names = list(self.current_signal.flags.keys())

        #Build dialogue window
        dialogue = tk.Toplevel(self)
        dialogue.title("Select Flags to Display")
        dialogue.geometry("400x500")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        #Main label
        main_label = ttk.Label(dialogue, text = "Select flags to display:")
        main_label.pack(pady=10)

        #Create frame for checkboxes with scrollbar
        checkbox_frame_outer = ttk.Frame(dialogue)
        checkbox_frame_outer.pack(fill=tk.BOTH, expand=True, padx=10,pady=10)

        canvas = tk.Canvas(checkbox_frame_outer)
        scrollbar = ttk.Scrollbar(checkbox_frame_outer, orient="vertical", command=canvas.yview)
        checkbox_frame = ttk.Frame(canvas)

        checkbox_frame.bind("<Configure>",lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0,0), window = checkbox_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left",fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        #Dict for checkbox variables
        checkbox_vars = {}

        #Create checkboxes for each flag
        for idx, name in enumerate(flag_names):
            visible = self.current_signal._flag_visibility.get(name, True)
            var = tk.BooleanVar(value = visible)
            checkbox_vars[name] = var
            cb = ttk.Checkbutton(checkbox_frame, text=name, variable=var)
            cb.grid(row=idx, column=0, sticky="w",padx=5,pady=2)
        
        def submit():
            for name, var in checkbox_vars.items():
                self.current_signal._flag_visibility[name] = var.get()
            dialogue.destroy()
            self.update_display(time_series=self.view_timeseries)
        
        #Select all/deselect all buttons
        button_frame = ttk.Frame(dialogue)
        button_frame.pack(pady=(0, 5))
        def select_all():
            for var in checkbox_vars.values():
                var.set(True)
        
        def deselect_all():
            for var in checkbox_vars.values():
                var.set(False)
        ttk.Button(button_frame, text="Select all", command=select_all).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Deselect all", command=deselect_all).pack(side="left", padx=5)
        
        #Submit button
        submit_button = ttk.Button(dialogue, text="Submit", command=submit)
        submit_button.pack(pady=20)

        dialogue.wait_window()
    
    def zoom_cmd(self):
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You need to build or import a signal before you can make changes to it.")
            return
        
        ret_var = None
        
        def validate_times():
            """Check if both fields are filled and values are within valid range"""
            try:
                start = float(start_var.get())
                end = float(end_var.get())
                
                min_time = self.current_signal.times[0]
                max_time = self.current_signal.times[-1]
                
                if start >= min_time and end <= max_time and start < end:
                    submit_button.config(state='normal')
                    return True
                else:
                    submit_button.config(state='disabled')
                    return False
            except ValueError:
                submit_button.config(state='disabled')
                return False
        
        def submit():
            nonlocal ret_var
            if validate_times():
                start_time = float(start_var.get())
                end_time = float(end_var.get())
                ret_var = [start_time, end_time]
                dialogue.destroy()
        
        # Create dialogue
        dialogue = tk.Toplevel(self)
        dialogue.title("Zoom to Time Range")
        dialogue.geometry("400x300")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()
        
        # Main label
        main_label = ttk.Label(dialogue, text=f"What would you like the start and end time of the signal to be? Please keep values between {self.current_signal.times[0]:.3f} and {self.current_signal.times[-1]:.3f} secs.")
        main_label.pack(pady=10)
        
        # Frame for entries
        entry_frame = ttk.Frame(dialogue)
        entry_frame.pack(pady=20)
        
        # Start time
        ttk.Label(entry_frame, text="Start time (secs):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        start_var = tk.StringVar()
        start_var.trace_add('write', lambda *args: validate_times())
        start_entry = ttk.Entry(entry_frame, textvariable=start_var, width=15)
        start_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # End time
        ttk.Label(entry_frame, text="End time (secs):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        end_var = tk.StringVar()
        end_var.trace_add('write', lambda *args: validate_times())
        end_entry = ttk.Entry(entry_frame, textvariable=end_var, width=15)
        end_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Valid range label
        min_time = self.current_signal.times[0]
        max_time = self.current_signal.times[-1]
        range_label = ttk.Label(entry_frame, 
                                text=f"Valid range: {min_time:.3f} to {max_time:.3f} seconds",
                                font=('TkDefaultFont', 8, 'italic'))
        range_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Submit button (disabled initially)
        submit_button = ttk.Button(dialogue, text="Submit", command=submit, state='disabled')
        submit_button.pack(pady=10)
        
        dialogue.wait_window()
        
        if ret_var is not None:
            self.display_time_lims = ret_var
            self.update_display()

    def view_ts_cmd(self):
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You need to build or import a signal before you can make changes to it.")
            return
        
        my_ts = self.current_signal.time_series
        if len(my_ts) == 0:
            gui_utilities.simple_dialogue("No time series available to display.")
            return
            
        names = [ts.name for ts in my_ts]
        
        # Create dialogue window
        dialogue = tk.Toplevel(self)
        dialogue.title("Select Time Series to View")
        dialogue.geometry("400x500")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()
        
        # Main label
        main_label = ttk.Label(dialogue, text="Select time series to display:")
        main_label.pack(pady=10)
        
        # Create frame for checkboxes with scrollbar
        checkbox_frame_outer = ttk.Frame(dialogue)
        checkbox_frame_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(checkbox_frame_outer)
        scrollbar = ttk.Scrollbar(checkbox_frame_outer, orient="vertical", command=canvas.yview)
        checkbox_frame = ttk.Frame(canvas)
        
        checkbox_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Dictionary to store checkbox variables
        checkbox_vars = {}
        
        # Create checkboxes for each time series
        for idx, name in enumerate(names):
            var = tk.BooleanVar(value=(name in self.view_timeseries))
            checkbox_vars[name] = var
            
            cb = ttk.Checkbutton(checkbox_frame, text=name, variable=var)
            cb.grid(row=idx, column=0, sticky='w', padx=5, pady=2)
        
        def submit():
            # Get list of selected time series names
            self.view_timeseries = [name for name, var in checkbox_vars.items() if var.get()]
            dialogue.destroy()
            self.update_display(time_series=self.view_timeseries)
        
        # Submit button
        submit_button = ttk.Button(dialogue, text="Submit", command=submit)
        submit_button.pack(pady=20)
        
        dialogue.wait_window()
    
    def alter_signal_cmd(self):
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You need to build or import a signal before you can make changes to it.")
            return
        if self.min_clean_time is None:
            self.min_clean_time = float(gui_utilities.text_entry("What is the minimum amount of consecutive clean signal you want to be considered for processing an analysis?"))
        fxn, lims = self.get_functions()
        self.current_signal = fxn.apply(self.current_signal, time_range=lims, flags_bool=True, min_clean_length=self.min_clean_time)
        self.current_signal.log_text(f"Applied function {fxn.name} with specs {fxn.args_dict} on time range {lims[0]:.3f}-{lims[1]:.3f} sec.")
        #Now, we need to re-run all our analyses on the new data
        analyzers = eeg_analyzers.AllAnalyzers._analyzers
        for analyzer in analyzers:
            if analyzer.name in [ts.name for ts in self.current_signal.time_series]:
                self.analyze_signal_cmd(analyzer_choice=analyzer) #Deletes old time series and recalculates on new signal

        self.update_display()
    
    def analyze_signal_cmd(self, analyzer_choice = None):
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You need to build or import a signal before you can make changes to it.")
            return
        if self.min_clean_time is None:
            self.min_clean_time = float(gui_utilities.text_entry("What is the minimum amount of consecutive clean signal you want to be considered for processing an analysis?"))
        
        # Calculate clean segments if min_clean_time is set
        clean_segments = None
        if self.min_clean_time > 0:
            from eeg_theme_park.utils.pipeline import find_clean_segments
            clean_segments = {}
            for ch_name in self.current_signal.all_channel_labels:
                ch_data = self.current_signal.all_channel_data[ch_name]
                clean_segments[ch_name] = find_clean_segments(
                    ch_data,
                    self.current_signal.srate,
                    self.min_clean_time
                )
        
        if analyzer_choice is None:
            analyzers = self.get_analyzer()
            for analyzer in analyzers:
                params = analyzer.get_params(eeg_object=self.current_signal, parent=self)
                if params is None:
                    continue  # User cancelled channel selection
                analyzer_inst = analyzer(**params)
                self.current_signal.time_series = [ts for ts in self.current_signal.time_series if ts.name != analyzer_inst.name]
                analyzer_inst.apply(self.current_signal, clean_segments=clean_segments)
        else:
            params = analyzer_choice.get_params(eeg_object=self.current_signal, parent=self)
            if params is None:
                return  # User cancelled
            analyzer_inst = analyzer_choice(**params)
            self.current_signal.time_series = [ts for ts in self.current_signal.time_series if ts.name != analyzer_inst.name]
            analyzer_inst.apply(self.current_signal, clean_segments=clean_segments)
            
        self.update_display()

    def build_signal_cmd(self):
        if not self.current_signal is None: #Make sure we don't overwrite a currently used signal if it has not been saved yet
            if self.has_unsaved_changes:
                save_current_signal = gui_utilities.yes_no("Your current signal has changes that haven't been saved. Would you like to save them before loading a new file?")
                if save_current_signal:
                    self.save_signal_cmd()
        self.current_signal = file_commands.build_signal_object()
        #NB: no log text here because logging is done automatically on EEGSignal object initialization
        self.signal_dir = None #None assigned, since we haven't saved yet
        self.has_unsaved_changes = True
        self.update_display()
    
    def rename_cmd(self, rename_saved = True):
        """
        Command to rename a given signal.

        Inputs:
        - rename_saved (bool): if True, will also rename the currently saved file. Setting to False acts like a save as, setting to True truly just renames the file (note, the process of renaming does NOT save current changes)
        """
        if self.current_signal is None:
            gui_utilities.simple_dialogue("There is no signal loaded. Please generate or load a signal to rename.")
            return  # Exit the function without returning anything
        old_name = copy.deepcopy(self.current_signal.name)
        #rename_saved False: save as functionality.
        new_name = gui_utilities.text_entry("What would you like to rename the file?")
        if new_name in ["", "."]:
            gui_utilities.simple_dialogue("Could not rename; please try again, but inputting a name.")
            return
        if self.signal_dir is None or rename_saved == False:
            self.current_signal.name = new_name
            self.save_signal_cmd() #Just a note: if the user stubbornly refuses to change the name to one that does not already exist, this could in theory become an infinite loop
            self.has_unsaved_changes = False
        else: #rename_saved True: will change the current signal name, and the name of the existing file WITHOUT overwriting the data
            if not self.signal_path.exists(): #No old file to rename; this may happen if a signal is built but not saved before the rename function is called
                self.current_signal.name = new_name
                self.save_signal_cmd() #This will just save the renamed file
            else:
                #Loads, renames and re-saves old signal with the new name, then updates the name of the current signal
                old_obj, old_path = file_commands.load_signal(file_path=self.signal_path)
                old_obj.name = new_name
                old_path_renamed = old_path.parent / f"{new_name}{old_path.suffix}"
                if old_path_renamed.exists():
                    continue_bool = gui_utilities.yes_no("There is already a file by this name in this directory. Would you like to continue, overwriting the old file? Yes to continue, no to enter a different name.")
                    if continue_bool:
                        file_commands.save_signal(old_obj,old_path_renamed)
                    if old_path != old_path_renamed:
                        old_path.unlink()
                    else:
                        self.rename_cmd(rename_saved=rename_saved)
                        return
                else:
                    file_commands.save_signal(eeg_signal_obj=old_obj,file_path=old_path_renamed)
                    old_path.unlink()
                    self.current_signal.name = new_name
        self.current_signal.log_text(f"Signal name changed from {old_name} to {self.current_signal.name}.")
        self.update_display()

    def save_signal_cmd(self):
        if self.has_unsaved_changes == False:
            pass #Nothing to save
        if self.current_signal is None:
            gui_utilities.simple_dialogue("You don't have a signal created or loaded. Please load or create a signal before attempting to save.")
        else: #I.e., we have a signal
            if self.signal_dir is None:
                save_dir = Path(gui_utilities.choose_dir())
                self.signal_dir = save_dir
            if self.signal_path.exists(): #I.e., we already have something by that name
                overwrite_bool = gui_utilities.yes_no(f"A file with that name at that path ({Path(self.signal_path)}) already exists. Would you like to overwrite? Answering yes will overwrite the current file, while answering no will take you to rename the current file before saving.")
                if overwrite_bool: #Simple overwrite
                    file_commands.save_signal(self.current_signal, Path(self.signal_path))
                    self.has_unsaved_changes = False
                else:
                    self.rename_cmd(rename_saved=False)
                    self.has_unsaved_changes = False
            else:
                file_commands.save_signal(self.current_signal, Path(self.signal_path))
                self.has_unsaved_changes = False
        if not self.current_signal is None:
            self.current_signal.log_text(f"Signal saved to {self.signal_path}")
        self.update_display()

    def load_signal_cmd(self, load_path: object = None):
        """
        Helper function to load an EEG file

        Inputs:
        - load_path (Path object): path from which we would like to load the file. If no path specified, will default to a prompt to allow the user to choose the file (or files). If no file selected, will simply do nothing.

        Outputs:
        None
        """
        self.view_timeseries=[]
        if not self.current_signal is None: #Make sure we don't overwrite a currently used signal if it has not been saved yet
            if self.has_unsaved_changes:
                save_current_signal = gui_utilities.yes_no("Your current signal has changes that haven't been saved. Would you like to save them before loading a new file?")
                if save_current_signal:
                    self.save_signal_cmd()
        eeg_signal_obj, file_path = file_commands.load_signal(load_path)
        if eeg_signal_obj is None or file_path is None:
            pass #Do nothing
        else:
            self.current_signal = eeg_signal_obj
            self.has_unsaved_changes = False
            self.signal_dir = Path(file_path.parent)
            self.display_time_lims = [self.current_signal.times[0], self.current_signal.times[-1]]
            self.analyze_time_lims = self.display_time_lims
        if not self.current_signal is None:
            self.current_signal.log_text("Signal loaded from file.")
        self.update_display()
     
    def get_analyzer(self, parent = None):
        """
        This function creates a window listing all Analyzers (signal processing functions) in eeg_analyzers.AllAnalyzers._analyzers, and returns the EEGAnalyzer subclasses corresponding to the ones the user selects.

        Inputs:
        - parent (Tkinter object): Tkinter parent window

        Outputs:
        - ret_var (list of EEGAnalyzer subclasses): list of all EEGAnalyzer subclasses to be used to process/analyze the signal. 
        """
        ret_var = None

        analyzers = eeg_analyzers.AllAnalyzers._analyzers
        names = [analyzer.name for analyzer in analyzers]

        # Create dialogue window
        dialogue = tk.Toplevel(parent)
        dialogue.title("Select Analyzers")
        dialogue.geometry("600x700")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        # Main label
        main_label = ttk.Label(dialogue, text="Choose your analyzers:")
        main_label.pack(pady=10)

        # Create frame for checkboxes with scrollbar
        checkbox_frame_outer = ttk.Frame(dialogue)
        checkbox_frame_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create canvas and scrollbar
        canvas = tk.Canvas(checkbox_frame_outer)
        scrollbar = ttk.Scrollbar(checkbox_frame_outer, orient="vertical", command=canvas.yview)
        checkbox_frame = ttk.Frame(canvas)

        checkbox_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Dictionary to store checkbox variables
        checkbox_vars = {}

        def check_selection(*args):
            """Enable submit button if at least one checkbox is selected"""
            any_selected = any(var.get() for var in checkbox_vars.values())
            submit_button.config(state='normal' if any_selected else 'disabled')

        # Create checkboxes for each analyzer
        for idx, name in enumerate(names):
            # Format the display name
            display_name = name[0].upper() + name[1:].lower() if name else name
            
            # Create checkbox variable
            var = tk.BooleanVar(value=False)
            checkbox_vars[name] = var  # Store with original name as key
            
            # Create checkbox
            cb = ttk.Checkbutton(checkbox_frame, text=display_name, variable=var, command=check_selection)
            cb.grid(row=idx, column=0, sticky='w', padx=5, pady=2)

        def submit():
            nonlocal ret_var
            # Get list of selected analyzer names (original formatting)
            selected_names = [name for name, var in checkbox_vars.items() if var.get()]
            
            # Return the analyzer objects that match the selected names
            ret_var = [analyzer for analyzer in analyzers if analyzer.name in selected_names]
            dialogue.destroy()

        # Submit button (disabled initially)
        submit_button = ttk.Button(dialogue, text="Submit", command=submit, state='disabled')
        submit_button.pack(pady=20)

        dialogue.wait_window()
        return ret_var
    
    def get_functions(self, parent = None):
        """
        This function creates a window listing all functions in signal_functions.AllFunctions._functions. 
        It also allows the user to select time limits for the function application.

        Inputs:
        - parent: Tkinter parent window

        Output:
        - ret_var (tuple): (EEGFunction subclass, [start_lim, end_lim]) or None if cancelled
        """
        ret_var = None
        
        # Flag to prevent circular updates
        updating = {'flag': False}
        
        # Variables created early
        start_time_var = None
        end_time_var = None
        start_scale_var = None
        end_scale_var = None
        slider = None
        
        def validate_time_limits():
            """Check if time limit entries are valid and within range"""
            if start_time_var is None or end_time_var is None:
                return False
            try:
                start_val = float(start_time_var.get())
                end_val = float(end_time_var.get())
                
                # Check if values are within valid range
                if (start_val >= self.current_signal.start_time and 
                    end_val <= self.current_signal.end_time and 
                    start_val < end_val):
                    return True
            except ValueError:
                pass
            return False
        
        def check_can_submit(*args):
            """Enable submit button only if function is selected and time limits are valid"""
            if listbox.curselection() and validate_time_limits():
                submit_button.config(state='normal')
            else:
                submit_button.config(state='disabled')
        
        def on_scale_change(*args):
            """Update entry boxes when scales change"""
            if updating['flag']:
                return
            updating['flag'] = True
            try:
                start_time_var.set(f"{start_scale_var.get():.3f}")
                end_time_var.set(f"{end_scale_var.get():.3f}")
                check_can_submit()
            finally:
                updating['flag'] = False
        
        def on_entry_change(*args):
            """Update scales when entry boxes change (if valid)"""
            if updating['flag']:
                return
            updating['flag'] = True
            try:
                start_val = float(start_time_var.get())
                end_val = float(end_time_var.get())
                
                # Only update scales if values are within range
                if (start_val >= self.current_signal.start_time and 
                    end_val <= self.current_signal.end_time and 
                    start_val < end_val):
                    start_scale_var.set(start_val)
                    end_scale_var.set(end_val)
            except ValueError:
                pass
            finally:
                check_can_submit()
                updating['flag'] = False
        
        def submit():
            nonlocal ret_var
            selection = listbox.curselection()
            if selection and validate_time_limits():
                selected_index = selection[0]
                selected_fxn_class = signal_functions.AllFunctions._functions[selected_index]
                params = selected_fxn_class.get_params(self.current_signal, parent=dialogue)
                if params is None:
                    gui_utilities.simple_dialogue("It seems like you quit adding parameters. Please try again, this time adding parameters.")
                    return

                selected_fxn = selected_fxn_class(**params)
                start_lim = float(start_time_var.get())
                end_lim = float(end_time_var.get())
                ret_var = (selected_fxn, [start_lim, end_lim])
            dialogue.destroy()
        
        all_fxn_names = signal_functions.AllFunctions.get_fxn_names()
        dialogue = tk.Toplevel(parent)
        dialogue.title("Signal functions")
        dialogue.geometry("700x750")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        # Main label
        main_label = ttk.Label(dialogue, text="Select a signal processing function:")
        main_label.pack(pady=10)

        # Create frame for listbox and scrollbar - FIXED HEIGHT
        list_frame = ttk.Frame(dialogue, height=150)
        list_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        list_frame.pack_propagate(False)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create listbox
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=listbox.yview)

        # Populate listbox with function names
        for name in all_fxn_names:
            listbox.insert(tk.END, name)
        
        # Time limits section
        time_frame = ttk.LabelFrame(dialogue, text="Select Time Range (seconds)", padding=10)
        time_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create variables FIRST
        start_time_var = tk.StringVar(value=f"{self.current_signal.start_time:.3f}")
        end_time_var = tk.StringVar(value=f"{self.current_signal.end_time:.3f}")
        start_scale_var = tk.DoubleVar(value=self.current_signal.start_time)
        end_scale_var = tk.DoubleVar(value=self.current_signal.end_time)

        # Create the scale widgets
        scale_frame = ttk.Frame(time_frame)
        scale_frame.pack(pady=10)

        start_scale = tk.Scale(scale_frame, 
                            from_=self.current_signal.start_time, 
                            to=self.current_signal.end_time, 
                            orient=tk.HORIZONTAL,
                            resolution=0.001, 
                            length=500, 
                            label="Start Time (s)",
                            variable=start_scale_var,
                            command=on_scale_change)
        start_scale.pack(pady=5)

        end_scale = tk.Scale(scale_frame, 
                            from_=self.current_signal.start_time, 
                            to=self.current_signal.end_time, 
                            orient=tk.HORIZONTAL,
                            resolution=0.001, 
                            length=500, 
                            label="End Time (s)",
                            variable=end_scale_var,
                            command=on_scale_change)
        end_scale.pack(pady=5)

        slider = None

        # Entry boxes for precise time specification
        entry_frame = ttk.Frame(time_frame)
        entry_frame.pack(pady=10)

        ttk.Label(entry_frame, text="Start Time (s):").grid(row=0, column=0, padx=5, sticky='e')
        start_entry = ttk.Entry(entry_frame, textvariable=start_time_var, width=15)
        start_entry.grid(row=0, column=1, padx=5)

        ttk.Label(entry_frame, text="End Time (s):").grid(row=1, column=0, padx=5, sticky='e')
        end_entry = ttk.Entry(entry_frame, textvariable=end_time_var, width=15)
        end_entry.grid(row=1, column=1, padx=5)

        # Range info label
        range_label = ttk.Label(entry_frame, 
                            text=f"Valid range: {self.current_signal.start_time:.3f} to {self.current_signal.end_time:.3f} seconds",
                            font=('TkDefaultFont', 8, 'italic'))
        range_label.grid(row=2, column=0, columnspan=2, pady=5)

        # Submit button (disabled initially)
        submit_button = ttk.Button(dialogue, text="Submit", command=submit, state='disabled')
        submit_button.pack(pady=20)

        # Bind events - use focusout for entries to avoid constant triggering
        listbox.bind('<<ListboxSelect>>', lambda e: check_can_submit())
        start_entry.bind('<FocusOut>', on_entry_change)
        start_entry.bind('<Return>', on_entry_change)
        end_entry.bind('<FocusOut>', on_entry_change)
        end_entry.bind('<Return>', on_entry_change)
        
        # Also trace for validation checking (but not for updating scales)
        start_time_var.trace_add('write', lambda *args: check_can_submit())
        end_time_var.trace_add('write', lambda *args: check_can_submit())

        dialogue.wait_window()
        return ret_var
    
    def _update_display(self, time_series = []):
        """
        Refresh the signal display window (the main window will have one row for the raw EEG signal as well as one row for each timeseries variable we want to show).

        Inputs:
        - timeseries (list): time series we would like to display along with our signal. Can take in strings (which will be matched to TimeSeries.main in eeg_signal_object.time_series) or integers (which will refer to indices in eeg_signal_object.time_series; the most recently calculated time series is always added to the end!)
        """
        if self.current_signal is None:
            return
        if len(time_series) == 0:
            time_series = self.view_timeseries
        #Quality control checks
        if not len(self.display_time_lims) in (0,2):
            raise ValueError(f"The time_lims variable should be either empty (displaying the whole signal) or contain two values (the start and end time). self.display_time_lims is currently contains {self.display_time_lims}")
        #Ensure all entries in time_series are either floats/ints or strings; there can be no mixing
        time_series_types = [type(item) for item in time_series]
        if len(set(time_series_types))>1:
            raise TypeError(f"time_series contained multiple types: {time_series_types}; please revise so that it contains only one type.")
        if set(time_series_types) == {str}: #All items are strings
            time_series = [ts for ts in self.current_signal.time_series if ts.name in time_series] #time_series now contains TimeSeries objects
        else:
            # Attempt to convert all items to integers and filter based on length of the current signal's time series
            try:
                time_series = [int(item) for item in time_series]
                time_series = [self.current_signal.time_series[i] for i in time_series if i<(len(self.current_signal.time_series)-1)] #time_series now contains TimeSeries objects
            except (ValueError, TypeError):
                # Create error message with each entry and its type
                type_pairs = [f"{item}: {type(item).__name__}" for item in time_series]
                error_msg = (f"time_series only accepts strings or integers. Please review time_series and its types: {(type_pairs)})")
                raise ValueError(error_msg)
        
        #Ensure our time series' data integrity
        EEG_data = self.current_signal.data
        EEG_times = self.current_signal.times
        if not(len(EEG_data)==len(EEG_times)):
            raise ValueError(f"For the signal, the number of data points and the number of time points should be the same; however, you have {len(EEG_data)} data points and {len(EEG_times)} time points.")
        for ts in time_series:
            ts_data_len = len(ts.values)
            ts_times_len = len(ts.times)
            if not(ts_data_len == ts_times_len):
                raise ValueError(f"For the time series, the number of data points and the number of time points should be the same; however, you have {ts_data_len} data points and {ts_times_len} time points in the time series {ts.name}.")
        
        # Clear previous canvas and toolbar if they exist
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()        
        
        #Start building the display figure
        nrows = 1+len(time_series) #One row for the EEG, len(time_series) rows for each time series we want to plot concommitantly
        fig, axes = plt.subplots(nrows=nrows,ncols=1) #axes is a tuple of Axes; each entry is one Axes (subplot)
        # Ensure axes is always iterable (subplots returns single Axes if nrows=1)
        if nrows == 1:
            axes = [axes]
        
        # Determine global x-axis limits from all data
        all_times = [EEG_times]
        for ts in time_series:
            all_times.append(ts.times)
        
        # Flatten all times and find min/max
        all_times_flat = [t for times in all_times for t in times]
        global_min_time = min(all_times_flat)
        global_max_time = max(all_times_flat)
        
        # Apply time_lims filtering if specified
        if len(self.display_time_lims) == 2:
            start_time = self.display_time_lims[0]
            end_time = self.display_time_lims[1]
            global_min_time = max(global_min_time, start_time)
            global_max_time = min(global_max_time, end_time)

        # Helper function to downsample data intelligently
        def downsample_data(times, data, max_points=1000):
            """
            Downsample data if it exceeds max_points while preserving visual features.
            Uses a min-max decimation approach to preserve peaks and troughs.
            """
            if len(data) <= max_points:
                print(f"Skipping downsample - already small enough: {len(data)} points.")
                return times, data
            
            # Calculate decimation factor
            factor = max(1, len(data) // (max_points // 2))
            print(f"Downsample factor: {factor}")
            
            # Decimate by taking min/max pairs in each window to preserve features
            downsampled_times = []
            downsampled_data = []
            
            for i in range(0, len(data), factor):
                window_end = min(i + factor, len(data))
                window_data = data[i:window_end]
                window_times = times[i:window_end]
                
                if len(window_data) > 0:
                    # Find min and max in this window
                    min_idx = np.argmin(window_data)
                    max_idx = np.argmax(window_data)
                    
                    # Add both min and max points (in time order)
                    if min_idx < max_idx:
                        downsampled_times.extend([window_times[min_idx], window_times[max_idx]])
                        downsampled_data.extend([window_data[min_idx], window_data[max_idx]])
                    else:
                        downsampled_times.extend([window_times[max_idx], window_times[min_idx]])
                        downsampled_data.extend([window_data[max_idx], window_data[min_idx]])
            
            print(f"Output length: {len(downsampled_data)}")
            return downsampled_times, downsampled_data
        
        #Helper function to insert nan values into TimeSeries visualizations based on srate
        def _insert_gap_nans(times, values, srate):
            """
            Detect temporal gaps in a time series and insert NaN values so that
            matplotlib breaks the plotted line instead of connecting across gaps.
            Uses the TimeSeries sampling rate (derived from the analyzer's advance_time)
            to determine the expected step size.
            """
            if len(times) < 2:
                return times, values
            
            times = np.asarray(times, dtype=float)
            values = np.asarray(values, dtype=float)
            
            expected_step = 1.0 / srate
            gap_threshold = 1.5 * expected_step
            
            diffs = np.diff(times)
            gap_indices = np.where(diffs > gap_threshold)[0]
            
            if len(gap_indices) == 0:
                return times.tolist(), values.tolist()
            
            new_times = []
            new_values = []
            prev = 0
            for gi in gap_indices:
                new_times.extend(times[prev:gi + 1])
                new_values.extend(values[prev:gi + 1])
                # Insert a NaN point midway through the gap
                new_times.append((times[gi] + times[gi + 1]) / 2)
                new_values.append(np.nan)
                prev = gi + 1
            new_times.extend(times[prev:])
            new_values.extend(values[prev:])
            
            return new_times, new_values
        
        # Filter data based on time_lims
        if len(self.display_time_lims) == 0: #Display entire signal if no time_lims specified
            plot_times = EEG_times
            plot_data = EEG_data
        else:
            # Filter to time_lims range
            start_time = self.display_time_lims[0]
            end_time = self.display_time_lims[1]
            mask = [(t >= start_time and t <= end_time) for t in EEG_times]
            plot_times = [t for t, m in zip(EEG_times, mask) if m]
            plot_data = [d for d, m in zip(EEG_data, mask) if m]
        
        # Downsample EEG data if needed
        plot_times, plot_data = downsample_data(plot_times, plot_data)
        
        # Plot EEG signal on first axis
        axes[0].plot(plot_times, plot_data, linewidth=0.5)
        axes[0].set_ylabel('Amplitude (uV)')
        axes[0].set_title(f'EEG Signal: {self.current_signal.name}')
        axes[0].grid(True)
        axes[0].set_xlim(global_min_time, global_max_time)
        
        # Format y-axis to avoid scientific notation
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axes[0].yaxis.set_major_formatter(formatter)
        
        # Plot each time series on subsequent axes
        for idx, ts in enumerate(time_series, start=1):
            if len(self.display_time_lims) == 0:
                ts_plot_times = ts.times
                ts_plot_values = ts.values
            else:
                start_time = self.display_time_lims[0]
                end_time = self.display_time_lims[1]
                ts_mask = [(t >= start_time and t <= end_time) for t in ts.times]
                ts_plot_times = [t for t, m in zip(ts.times, ts_mask) if m]
                ts_plot_values = [v for v, m in zip(ts.values, ts_mask) if m]
            
            #Downsample timeseries if needed
            ts_plot_times, ts_plot_values = downsample_data(ts_plot_times, ts_plot_values)
            
            # Insert NaNs at temporal gaps so matplotlib breaks the line
            # (analyzers skip noisy windows entirely, leaving gaps rather than NaN placeholders)
            ts_plot_times, ts_plot_values = _insert_gap_nans(ts_plot_times, ts_plot_values, ts.srate)
            
            # Use different color for each time series
            color = f'C{idx}'  # matplotlib color cycle (C0, C1, C2, etc.)
            axes[idx].plot(ts_plot_times, ts_plot_values, color=color, linewidth=0.5)
            
            # Use different color for each time series
            color = f'C{idx}'  # matplotlib color cycle (C0, C1, C2, etc.)
            axes[idx].plot(ts_plot_times, ts_plot_values, color=color, linewidth=0.5)
            # Set y-axis label with units if available
            ylabel = f"{ts.print_name} ({ts.units})" if hasattr(ts, 'units') and ts.units else ts.name
            axes[idx].set_ylabel(ylabel)
            axes[idx].grid(True)
            axes[idx].set_xlim(global_min_time, global_max_time)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axes[idx].yaxis.set_major_formatter(formatter)
        
        # Add x-label only to bottom plot
        axes[-1].set_xlabel('Time (s)')
        
        # Plot flags if they exist
        if len(self.current_signal.flags) > 0:
            flags = self.current_signal.flags
            flag_colors = {}  # Store colors for legend
            color_idx = len(time_series) + 1  # Start after time series colors
            for flag_name, flag_value in flags.items():
                #Do not display flags marked by user as non-visible
                if not hasattr(self.current_signal, '_flag_visibility'):
                    self.current_signal._flag_visibility = {}
                if not self.current_signal._flag_visibility.get(flag_name, True):
                    continue
                # Assign unique color to this flag
                color = f'C{color_idx}'
                flag_colors[flag_name] = color
                color_idx += 1
                if len(flag_value) == 1:
                    # Single vertical line at flag_value[0]
                    for ax in axes:
                        ax.axvline(x=flag_value[0], color=color, linestyle='--', linewidth=1.5, label=flag_name)
                elif len(flag_value) == 3:
                    # Two vertical lines at flag_value[0] and flag_value[1]
                    for ax in axes:
                        ax.axvline(x=flag_value[0], color=color, linestyle='--', linewidth=1.5)
                        ax.axvline(x=flag_value[1], color=color, linestyle='--', linewidth=1.5, label=flag_name)
                        # Shade between lines if flag_value[2] is True
                        if flag_value[2]:
                            ax.axvspan(flag_value[0], flag_value[1], alpha=0.2, color=color)
            # Add legend to first axis only (to avoid clutter)
            if flag_colors:
                axes[0].legend(loc='upper right', fontsize='small')

        # Create canvas and embed in tkinter window
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()    
    #------------------------------------