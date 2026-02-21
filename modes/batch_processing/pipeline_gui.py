"""
File that handles creating the GUI for building a Pipeline object.
"""

from eeg_theme_park.utils import gui_utilities, eeg_analyzers, signal_functions, pipeline
import copy
import tkinter as tk
from tkinter import ttk
import inspect
import pickle
from tkinter import filedialog

def build_pipeline(available_channels=None):
    """
    Function that opens up a GUI for the user to build their own Pipeline object. Specifically, this will have three tkinter List Boxes, a >> button, a << button, and a submit button. The leftmost will be called "Current Pipeline" (CP), the middle will be called "Transform Signal" (TS), and the last will be called "Analyze Signal" (AS). The TS box will be populated by all names in fxn_names, while the AS box will be populated by all names in analyzer_names. Then, I want the user to be able to highlight a name in TS or AS and then, by clicking a << button, have it added to the bottom of the list of names in CP. CP will be represented by an array of arrays cp_entries, where each member is of the form [name, object]. Name should be the function name displayed, while object should be the instantiated version of the object (see below). 
     
    If an entry from TS is moved over, call get_params_gui() using the corresponding function's class. Take the returned dictionary and use it to initialize an instance of the function. Finally, add this function name and instance to the end of cp_entries.
     
    The user should also be able to click on/highlight a name in the CP column, and click a >> button, which should delete that name from the CP column (and its entry). When the user hits Submit, the array fin_blocks should be created, which will contain, in order, all the objects in cp_entries. Finally, return Pipeline(fin_blocks)

    Output:
    - final_pipeline (Pipeline object): the final pipeline the user has created
    """
    
    # First, get minimum time from user
    min_time_str = gui_utilities.text_entry("What is the minimum amount of time (in seconds) you want a segment to have to be analyzable?")
    # Handle if user cancelled
    if min_time_str is None:
        return None
    # Convert to float
    try:
        min_time = float(min_time_str)
    except ValueError:
        gui_utilities.simple_dialogue("Invalid minimum time value. Please enter a number.")
        return None
    
    # Get all functions and analyzers
    all_functions = copy.deepcopy(signal_functions.AllFunctions._functions)
    fxn_names = [fxn.name for fxn in all_functions]
    if not len(fxn_names) == len(set(fxn_names)):
        raise ValueError(f"Some of your functions in signal_functions.py have identical names. This will break the pipeline building. Please change the function names before proceeding. Specifically, the function names you have are {fxn_names}.")
    
    all_analyzers = copy.deepcopy(eeg_analyzers.AllAnalyzers._analyzers)
    analyzer_names = [analyzer.name for analyzer in all_analyzers]
    if not len(analyzer_names) == len(set(analyzer_names)):
        raise ValueError(f"Some of your analyzers in eeg_analyzers.py have identical names. This will break the pipeline building. Please change the analyzer names before proceeding. Specifically, the analyzer names you have are {analyzer_names}.")
    
    # Create dictionaries for easy lookup
    fxn_dict = {fxn.name: fxn for fxn in all_functions}
    analyzer_dict = {analyzer.name: analyzer for analyzer in all_analyzers}
    full_dict = {**fxn_dict, **analyzer_dict}
    
    # Data structure to hold current pipeline
    cp_entries = []  # List of [name, instance] pairs
    
    # Variable to store final result
    final_pipeline = {'result': None}
    
    # Create main window
    window = tk.Toplevel()
    window.title("Build Pipeline")
    window.geometry("900x600")
    window.grab_set()
    window.lift()
    window.focus_force()
    
    # Main instruction label
    main_label = ttk.Label(window, text="Build your analysis pipeline by selecting functions and analyzers", font=('TkDefaultFont', 10, 'bold'))
    main_label.pack(pady=10)
    
    # Create main frame to hold the three listboxes and buttons
    main_frame = ttk.Frame(window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Configure grid weights for responsive layout
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=0)
    main_frame.columnconfigure(2, weight=1)
    main_frame.columnconfigure(3, weight=0)
    main_frame.columnconfigure(4, weight=1)
    main_frame.rowconfigure(1, weight=1)
    
    # --- Current Pipeline (CP) Section ---
    cp_label = ttk.Label(main_frame, text="Current Pipeline (CP)", font=('TkDefaultFont', 9, 'bold'))
    cp_label.grid(row=0, column=0, pady=5)
    
    cp_frame = ttk.Frame(main_frame)
    cp_frame.grid(row=1, column=0, sticky='nsew', padx=5)
    
    cp_scrollbar = ttk.Scrollbar(cp_frame)
    cp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    cp_listbox = tk.Listbox(cp_frame, yscrollcommand=cp_scrollbar.set, selectmode=tk.SINGLE)
    cp_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    cp_scrollbar.config(command=cp_listbox.yview)
    
    # --- Buttons between CP and TS/AS ---
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=1, column=1, padx=10)
    
    # << button (add to pipeline)
    add_button = ttk.Button(button_frame, text="<<", width=5)
    add_button.pack(pady=5)
    
    # >> button (remove from pipeline)
    remove_button = ttk.Button(button_frame, text=">>", width=5)
    remove_button.pack(pady=5)
    
    # --- Transform Signal (TS) Section ---
    ts_label = ttk.Label(main_frame, text="Transform Signal (TS)", font=('TkDefaultFont', 9, 'bold'))
    ts_label.grid(row=0, column=2, pady=5)
    
    ts_frame = ttk.Frame(main_frame)
    ts_frame.grid(row=1, column=2, sticky='nsew', padx=5)
    
    ts_scrollbar = ttk.Scrollbar(ts_frame)
    ts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    ts_listbox = tk.Listbox(ts_frame, yscrollcommand=ts_scrollbar.set, selectmode=tk.SINGLE)
    ts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ts_scrollbar.config(command=ts_listbox.yview)
    
    # Populate TS listbox
    for name in fxn_names:
        ts_listbox.insert(tk.END, name)
    
    # Spacer
    spacer_frame = ttk.Frame(main_frame, width=20)
    spacer_frame.grid(row=1, column=3)
    
    # --- Analyze Signal (AS) Section ---
    as_label = ttk.Label(main_frame, text="Analyze Signal (AS)", font=('TkDefaultFont', 9, 'bold'))
    as_label.grid(row=0, column=4, pady=5)
    
    as_frame = ttk.Frame(main_frame)
    as_frame.grid(row=1, column=4, sticky='nsew', padx=5)
    
    as_scrollbar = ttk.Scrollbar(as_frame)
    as_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    as_listbox = tk.Listbox(as_frame, yscrollcommand=as_scrollbar.set, selectmode=tk.SINGLE)
    as_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    as_scrollbar.config(command=as_listbox.yview)
    
    # Populate AS listbox
    for name in analyzer_names:
        as_listbox.insert(tk.END, name)
    
    # --- Button Functions ---
    
    def add_to_pipeline():
        """Add selected item from TS or AS to CP"""
        ts_selection = ts_listbox.curselection()
        as_selection = as_listbox.curselection()
        
        if ts_selection:
            idx = ts_selection[0]
            name = ts_listbox.get(idx)
            fxn_class = fxn_dict[name]
            
            params_dict = get_params_gui(fxn_class, window)
            if params_dict is None:
                return
            
            instance = fxn_class(**params_dict)
            cp_entries.append([name, instance])
            cp_listbox.insert(tk.END, name)
            
        elif as_selection:
            idx = as_selection[0]
            name = as_listbox.get(idx)
            analyzer_class = analyzer_dict[name]
            
            # Check if this is a MultiChannelAnalyzer that needs channel selection
            if issubclass(analyzer_class, eeg_analyzers.MultiChannelAnalyzer):
                if available_channels is not None and len(available_channels) > 0:
                    # Create a lightweight stand-in so get_params can use its dropdown path
                    class _ChannelProxy:
                        pass
                    proxy = _ChannelProxy()
                    proxy.all_channel_labels = available_channels
                    params = analyzer_class.get_params(eeg_object=proxy, parent=window)
                else:
                    # Fall back to text entry (get_params handles eeg_object=None)
                    params = analyzer_class.get_params(eeg_object=None, parent=window)
                
                if params is None:
                    return
                instance = analyzer_class(**params)
            else:
                # Regular (single-channel) analyzer — check if it needs params too
                params_dict = get_params_gui(analyzer_class, window)
                if params_dict is None:
                    return
                instance = analyzer_class(**params_dict)
            
            cp_entries.append([name, instance])
            cp_listbox.insert(tk.END, instance.display_name if hasattr(instance, 'display_name') else name)
        else:
            gui_utilities.simple_dialogue("Please select an item from Transform Signal or Analyze Signal first.")
    
    def remove_from_pipeline():
        """Remove selected item from CP"""
        selection = cp_listbox.curselection()
        
        if not selection:
            gui_utilities.simple_dialogue("Please select an item from Current Pipeline to remove.")
            return
        
        # Get the index
        idx = selection[0]
        
        # Remove from cp_entries and listbox
        del cp_entries[idx]
        cp_listbox.delete(idx)
    
    def submit_pipeline():
        """Create Pipeline object and close window"""
        if len(cp_entries) == 0:
            should_continue = gui_utilities.yes_no("Your pipeline is empty. Do you want to create an empty pipeline?")
            if not should_continue:
                return
        
        # Validate that no EEGFunction comes after an EEGAnalyzer
        seen_analyzer = False
        for name, instance in cp_entries:
            # Check if this is an analyzer
            if name in analyzer_dict:
                seen_analyzer = True
            # Check if this is a function but we've already seen an analyzer
            elif name in fxn_dict and seen_analyzer:
                gui_utilities.simple_dialogue("Invalid pipeline: Transform Signal functions cannot come after Analyze Signal analyzers. Please reorder your pipeline.")
                return
        
        # Extract just the instances (not names) from cp_entries
        fin_blocks = [entry[1] for entry in cp_entries]
        
        # Create Pipeline object
        final_pipeline['result'] = pipeline.Pipeline(fin_blocks, min_clean_length = min_time)
        
        window.destroy()
    
    # Configure button commands
    add_button.config(command=add_to_pipeline)
    remove_button.config(command=remove_from_pipeline)
    
    # --- Submit Button ---
    submit_button = ttk.Button(window, text="Submit", command=submit_pipeline)
    submit_button.pack(pady=20)
    
    # Wait for window to close
    window.wait_window()
    
    return final_pipeline['result']

def import_pipeline():
    """Opens a window to allow the user to select a pipeline .ppl file. When selected, unpickles the object and returns it for the user. """
    
    # Open file dialog to select .ppl file
    file_path = filedialog.askopenfilename(
        title="Select Pipeline File",
        filetypes=[("Pipeline files", "*.ppl"), ("All files", "*.*")],
        defaultextension=".ppl"
    )
    
    # If user cancelled, return None
    if not file_path:
        return None
    
    # Try to unpickle the pipeline object
    try:
        with open(file_path, 'rb') as f:
            imported_pipeline = pickle.load(f)
        
        # Verify it's a Pipeline object
        if not isinstance(imported_pipeline, pipeline.Pipeline):
            gui_utilities.simple_dialogue(f"Error: The selected file does not contain a valid Pipeline object.")
            return None
        
        gui_utilities.simple_dialogue(f"Pipeline successfully imported from:\n{file_path}")
        return imported_pipeline
    
    except Exception as e:
        gui_utilities.simple_dialogue(f"Error loading pipeline file:\n{str(e)}")
        return None

def get_params_gui(cls, parent=None):
    """
    Creates a dialogue to collect parameter values from the user based on the __init__() method's signature (as defined in the subclass).

    Inputs:
    - cls (class): class of the function we are initializing
    - parent (tk widget): parent window for the dialogue

    Outputs:
    - ret_var (dict): a dictionary of values in the format {"param_name":value}
    """
    sig = inspect.signature(cls.__init__)  # Accesses the subclass's __init__ signature
    params_to_collect = {name: param for name, param in sig.parameters.items() 
                        if name not in ('self', 'kwargs') and 
                        param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    
    if len(params_to_collect) == 0:
        return {}
    
    ret_var = {}
    string_vars = {}

    def check_fields(*args):
        """Ensures required fields (i.e., all params) are filled"""
        all_required_filled = True
        for param_name, param in params_to_collect.items():
            if param.default is inspect.Parameter.empty:  # No default value
                if string_vars[param_name].get().strip() == "":
                    all_required_filled = False
                    break
        
        submit_button.config(state='normal' if all_required_filled else 'disabled')

    def submit():
        nonlocal ret_var
        # Collect values from entry boxes
        for param_name, str_var in string_vars.items():
            value_str = str_var.get().strip()
            if value_str == "":
                # Check if this parameter has a default value
                param = params_to_collect[param_name]
                if param.default is inspect.Parameter.empty:
                    gui_utilities.simple_dialogue(f"Parameter '{param_name}' is required but was not provided.")
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
                    gui_utilities.simple_dialogue(f"Could not convert '{value_str}' to appropriate type for parameter '{param_name}'.")
                    return
        
        dialogue.destroy()

    # Build main box
    dialogue = tk.Toplevel(parent)
    dialogue.title(f"Parameters for {cls.name}")
    dialogue.geometry("600x400")
    dialogue.grab_set()
    dialogue.lift()
    dialogue.focus_force()
    
    main_label = ttk.Label(dialogue, text=f"Please enter parameters for '{cls.name}'", wraplength=580)
    main_label.grid(column=0, row=0, columnspan=2, padx=10, pady=10)

    row = 1
    
    # Get units dict if available
    units_dict = getattr(cls, 'params_units_dict', {})

    # Create entry boxes
    for param_name, param in params_to_collect.items():
        # Create label
        label_text = param_name.replace('_', ' ').title()
        if param_name in units_dict.keys():
            label_text += f" ({units_dict[param_name]})"
        label = ttk.Label(dialogue, text=f"{label_text}:")
        label.grid(column=0, row=row, sticky="w", padx=10, pady=5)
        
        # Create entry box
        str_var = tk.StringVar(dialogue)
        if param.default is not inspect.Parameter.empty:
            str_var.set(str(param.default))
        str_var.trace_add('write', check_fields)
        entry = ttk.Entry(dialogue, textvariable=str_var, width=30)
        entry.grid(column=1, row=row, padx=10, pady=5)
        string_vars[param_name] = str_var
        row += 1

    # Submit button
    submit_button = ttk.Button(dialogue, text="Submit", command=submit, state='disabled')
    submit_button.grid(column=0, columnspan=2, row=row, pady=20)

    # Initial check
    check_fields()
    
    dialogue.wait_window()
    return ret_var if ret_var else None