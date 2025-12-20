import tkinter as tk
from tkinter import ttk
from ctypes import windll
from tkinter import filedialog
import os
from pathlib import Path

def simple_dialogue(text: str, parent=None):
    """
    Module-level fxn to allow for display of a dialogue box for the user.
    Inputs:
    - text (str): Text to be displayed

    Outputs:
    None
    """
    dialogue = tk.Toplevel(parent) #When parent is None, this keeps the dialogue separate from other windows
    dialogue.geometry('600x600')
    # Make window always on top
    dialogue.attributes('-topmost', True)
    dialogue.lift()
    dialogue.focus_force()

    label = ttk.Label(dialogue, text=text, wraplength=580)
    label.pack(padx=10, pady=10, fill='both', expand=True)
    dialogue.grab_set()

def yes_no(text, parent=None) -> bool:
    """
    Creates a dialogue box that allows the user to select yes or no, and then returns the user's decision. Text is customaizable.

    Input:
    - text (str): text to display above the yes/no button
    - parent (tkinter root): parent window, if it exists

    Output:
    - choice (bool): True if user selects Yes, false if user selects No
    """
    choice = None
    def on_yes():
        nonlocal choice
        choice = True
        dialogue.destroy()

    def on_no():
        nonlocal choice
        choice = False
        dialogue.destroy()

    dialogue = tk.Toplevel(parent)
    dialogue.title("")
    dialogue.geometry("600x600")
    # Make window always on top
    dialogue.attributes('-topmost', True)
    dialogue.lift()
    dialogue.focus_force()

    label = ttk.Label(dialogue, text=text, wraplength=580)
    yes_btn = ttk.Button(dialogue, text = "Yes", command = on_yes)
    no_btn = ttk.Button(dialogue, text = "No", command = on_no)
    label.grid(row=0, column=0, columnspan=4, padx=10, pady=10, sticky='ew')
    yes_btn.grid(row=1, column=0, columnspan=2, pady=10)
    no_btn.grid(row=1, column=2, columnspan=2, pady=10)
    
    dialogue.grid_columnconfigure(0, weight=1)
    dialogue.grid_columnconfigure(1, weight=1)
    dialogue.grid_columnconfigure(2, weight=1)
    dialogue.grid_columnconfigure(3, weight=1)
    
    windll.shcore.SetProcessDpiAwareness(1)
    dialogue.grab_set()
    dialogue.wait_window()
    return choice

def text_entry(text: str, parent=None) -> str:
    """
    Creates a dialogue box with a text entry field and OK button.
    
    Inputs:
    - text (str): text directing the user what to do with the text entry box
    - parent (tkinter root): parent window, if it exists

    Outputs:
    - output_text (str): text that was in the text entry box, or None if cancelled
    """
    output_text = None
    
    def on_ok():
        nonlocal output_text
        output_text = entry.get()
        dialogue.destroy()
    
    dialogue = tk.Toplevel(parent)
    dialogue.title("")
    dialogue.geometry("600x200")

    # Make window always on top
    dialogue.attributes('-topmost', True)
    dialogue.lift()
    dialogue.focus_force()
    
    label = ttk.Label(dialogue, text=text, wraplength=580)
    entry = ttk.Entry(dialogue, width=50)
    ok_btn = ttk.Button(dialogue, text="OK", command=on_ok)
    
    label.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
    entry.grid(row=1, column=0, padx=10, pady=10)
    ok_btn.grid(row=2, column=0, padx=10, pady=10)
    
    dialogue.grid_columnconfigure(0, weight=1)
    
    windll.shcore.SetProcessDpiAwareness(1)
    dialogue.grab_set()
    dialogue.wait_window()
    
    return output_text

def choose_dir(text: str = "Select a directory:", parent=None) -> str:
    """
    Creates a dialogue box that allows the user to select a directory.
    
    Input:
    - text (str): text to display in the dialogue title
    - parent (tkinter root): parent window, if it exists
    
    Output:
    - directory_path (str): absolute path of the selected directory, or None if cancelled
    """
    
    dialogue = tk.Toplevel(parent)
    dialogue.withdraw()  # Hide the empty toplevel window
    dialogue.attributes('-topmost', True)
    
    directory_path = filedialog.askdirectory(
        parent=dialogue,
        title=text,
        initialdir=os.getcwd()
    )
    
    dialogue.destroy()
    return Path(directory_path) if directory_path else None

def choose_channel(channels_list, parent=None):
    """
    Function to help user choose the channel they want to import.

    Input:
    - channels_list (list of str): list containing all the channel names

    Output:
    - out_list (list): list in the form [a: str, b: int], where a is the name (in all uppercase) of the chosen channel and b is the index of a in channels_list
    """
    out_list = None
    selected_index = tk.IntVar(value=-1)
    
    def on_submit():
        nonlocal out_list
        idx = selected_index.get()
        if idx >= 0:
            out_list = [channels_list[idx].upper(), idx]
            dialogue.destroy()
    
    dialogue = tk.Toplevel(parent)
    dialogue.title("")
    dialogue.geometry("800x400")
    
    # Make window always on top
    dialogue.attributes('-topmost', True)
    dialogue.lift()
    dialogue.focus_force()
    
    label = ttk.Label(dialogue, text="Please choose the channel you want:", wraplength=780)
    label.pack(padx=10, pady=10)
    
    # Frame for radiobuttons with scrollbar
    frame = ttk.Frame(dialogue)
    frame.pack(padx=10, pady=10, fill='both', expand=True)
    
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Create radiobuttons for each channel
    for i, channel in enumerate(channels_list):
        rb = ttk.Radiobutton(
            scrollable_frame,
            text=channel.upper(),
            variable=selected_index,
            value=i
        )
        rb.pack(anchor='w', padx=5, pady=2, fill='x')
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    submit_btn = ttk.Button(dialogue, text="Submit", command=on_submit, state='disabled')
    submit_btn.pack(pady=10)
    
    # Enable submit button when selection is made
    def on_select(*args):
        if selected_index.get() >= 0:
            submit_btn.configure(state='normal')
    
    selected_index.trace('w', on_select)
    
    windll.shcore.SetProcessDpiAwareness(1)
    dialogue.grab_set()
    dialogue.wait_window()
    
    return out_list

def get_file(text: str = "Select a file:", file_types=None, parent=None) -> str:
    """
    Creates a dialogue box that allows the user to select a file.
    
    Input:
    - text (str): text to display in the dialogue title
    - file_types (list of tuples): file type filters in format [("Description", "*.ext")]
    - parent (tkinter root): parent window, if it exists
    
    Output:
    - file_path (Path): Path object of the selected file, or None if cancelled
    """
    
    dialogue = tk.Toplevel(parent)
    dialogue.withdraw()  # Hide the empty toplevel window
    dialogue.attributes('-topmost', True)
    
    if file_types is None:
        file_types = [("All files", "*.*")]
    
    file_path = filedialog.askopenfilename(
        parent=dialogue,
        title=text,
        initialdir=os.getcwd(),
        filetypes=file_types
    )
    
    dialogue.destroy()
    return Path(file_path) if file_path else None

def get_specs(signal_specs = {}, parent = None):
    """
    This function creates a Window in which the user can input the specs for their signal; it can have the signal_specs dictionary passed to it, in which case it will find which specs for the signal do exist in it and auto-populate them into the text boxes.
    
    Inputs:
    - signal_specs (dict): dict of the form {str : var} containing the specifications for the signal being built
    - parent (Tkinter root): used when get_specs is interfacing with the main dialogue; if None, TKinter will handle and create a new root to display the box when TopLevel() is called
    
    Outputs:
    - signal_specs_clean (dict): dict of the form {str: var} containing only valid dict keys and their values. Returned when Submit button is clicked
    """
    ret_var = {}
    
    def check_fields(*args):
        """Check if all fields are filled and enable/disable submit button"""
        all_filled = all([
            name_str.get().strip(),
            amp_str.get().strip(),
            freq_str.get().strip(),
            phase_str.get().strip(),
            srate_str.get().strip(),
            length_str.get().strip(),
            start_str.get().strip()
        ])
        submit_button.config(state='normal' if all_filled else 'disabled')
    
    def submit():
        nonlocal ret_var
        raw_values = {
            "name": name_str.get().strip(),
            "amp": amp_str.get().strip(),
            "freq": freq_str.get().strip(),
            "phase": phase_str.get().strip(),
            "srate": srate_str.get().strip(),
            "time_length": length_str.get().strip(),
            "start_time": start_str.get().strip()
        }


        if any([value is None for value in ret_var.values()]):
            dialogue.destroy() #Clear the old dialogue to make way for a new one
            simple_dialogue("Please fill out every required field before proceeding.")
            ret_var = get_specs(signal_specs=signal_specs,parent=parent)
            return
        else:
            ret_var = {
            "name": raw_values["name"],
            "amp": float(raw_values["amp"]),
            "freq": float(raw_values["freq"]),
            "phase": float(raw_values["phase"]),
            "srate": int(raw_values["srate"]),
            "time_length": float(raw_values["time_length"]),
            "start_time": float(raw_values["start_time"])
        }
            dialogue.destroy()
            return

    vars_dict = {
        "name":"name_str",
        "amp":"amp_str",
        "freq":"freq_str",
        "phase":"phase_str",
        "srate":"srate_str",
        "time_length":"length_str",
        "start_time":"start_str"
    }
    
    dialogue = tk.Toplevel(parent)
    dialogue.title("Signal specifications")
    dialogue.geometry("600x600")
    dialogue.grab_set()
    dialogue.lift()
    dialogue.focus_force()

    #Build text boxes and buttons ------------------
    main_label = ttk.Label(dialogue, text = "Please input the missing specification values to build your signal")
    main_label.grid(column = 0, row= 0, columnspan = 2)

    l1 = ttk.Label(dialogue, text = "Name: ")
    l1.grid(column = 0, row = 1, sticky="w")
    name_str = tk.StringVar(dialogue)
    name_str.trace_add('write', check_fields)
    e1 = ttk.Entry(dialogue, textvariable=name_str)
    e1.grid(column = 1, row = 1)

    l2 = ttk.Label(dialogue, text = "Amplitude (uV): ")
    l2.grid(column = 0, row = 2, sticky="w")
    amp_str = tk.StringVar(dialogue)
    amp_str.trace_add('write', check_fields)
    e2 = ttk.Entry(dialogue, textvariable=amp_str)
    e2.grid(column = 1, row = 2)

    l3 = ttk.Label(dialogue, text = "Frequency (Hz): ")
    l3.grid(column = 0, row = 3, sticky="w")
    freq_str = tk.StringVar(dialogue)
    freq_str.trace_add('write', check_fields)
    e3 = ttk.Entry(dialogue, textvariable=freq_str)
    e3.grid(column = 1, row = 3)

    l4 = ttk.Label(dialogue, text = "Phase (radian; pi/6 = 0.52, pi/4 = 0.79): ")
    l4.grid(column = 0, row = 4, sticky="w")
    phase_str = tk.StringVar(dialogue)
    phase_str.trace_add('write', check_fields)
    e4 = ttk.Entry(dialogue, textvariable=phase_str)
    e4.grid(column = 1, row = 4)

    l5 = ttk.Label(dialogue, text = "Sampling rate (Hz): ")
    l5.grid(column = 0, row = 5, sticky="w")
    srate_str = tk.StringVar(dialogue)
    srate_str.trace_add('write', check_fields)
    e5 = ttk.Entry(dialogue, textvariable=srate_str)
    e5.grid(column = 1, row = 5)

    l6 = ttk.Label(dialogue, text = "Signal length (sec): ")
    l6.grid(column = 0, row = 6, sticky="w")
    length_str = tk.StringVar(dialogue)
    length_str.trace_add('write', check_fields)
    e6 = ttk.Entry(dialogue, textvariable=length_str)
    e6.grid(column = 1, row = 6)

    l7 = ttk.Label(dialogue, text = "Start time (seconds from 0): ")
    l7.grid(column = 0, row = 7, sticky="w")
    start_str = tk.StringVar(dialogue)
    start_str.trace_add('write', check_fields)
    e7 = ttk.Entry(dialogue, textvariable=start_str)
    e7.grid(column = 1, row = 7)

    submit_button = ttk.Button(dialogue, text = "Submit", command = submit, state='disabled')
    submit_button.grid(column = 0, columnspan=2, row = 8)
    #End build text boxes and buttons ------------------

    #Populate text boxes--------
    if signal_specs is None:
        pass
    else:
        for key in signal_specs:
            if key in vars_dict: #It's a valid key
                var_name = vars_dict[key]
                locals()[var_name].set(signal_specs[key]) #Assigns our StringVar to the pre-included value in signal_specs
    #End populate text boxes------
    
    # Initial check to see if all fields are already populated
    check_fields()
    
    dialogue.wait_window()
    #dialogue.destroy() only occurs after hitting submit, which assigns ret_var to the cleaned specifications dictionary
    return ret_var

def dropdown_menu(text, options, multiple=False, parent=None):
    """
    Creates a dialogue box with a dropdown menu or checkboxes for selection.
    
    Inputs:
    - text (str): instruction text to display
    - options (list): list of options to display
    - multiple (bool): if True, allows multiple selections with checkboxes
    - parent (tkinter root): parent window, if it exists
    
    Outputs:
    - result: selected option(s) as string (single) or list (multiple), or None if cancelled
    """
    result = {'selection': None}
    
    def on_submit():
        if multiple:
            # Collect all checked items
            selected = [opt for opt, var in checkbox_vars.items() if var.get()]
            result['selection'] = selected if selected else None
        else:
            # Get single selection
            selection = selected_var.get()
            result['selection'] = selection if selection else None
        dialogue.destroy()
    
    def on_cancel():
        dialogue.destroy()
    
    dialogue = tk.Toplevel(parent)
    dialogue.title("Select Options")
    dialogue.geometry("600x400")
    dialogue.attributes('-topmost', True)
    dialogue.lift()
    dialogue.focus_force()
    
    # Instruction label
    label = ttk.Label(dialogue, text=text, wraplength=580)
    label.pack(padx=10, pady=10)
    
    # Create scrollable frame for options
    canvas_frame = ttk.Frame(dialogue)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    canvas = tk.Canvas(canvas_frame)
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    if multiple:
        # Create checkboxes for multiple selection
        checkbox_vars = {}
        for option in options:
            var = tk.BooleanVar(value=False)
            checkbox_vars[option] = var
            cb = ttk.Checkbutton(scrollable_frame, text=option, variable=var)
            cb.pack(anchor='w', padx=5, pady=2)
    else:
        # Create radiobuttons for single selection
        selected_var = tk.StringVar(value="")
        for option in options:
            rb = ttk.Radiobutton(scrollable_frame, text=option, 
                                variable=selected_var, value=option)
            rb.pack(anchor='w', padx=5, pady=2)
    
    # Button frame
    button_frame = ttk.Frame(dialogue)
    button_frame.pack(pady=10)
    
    submit_btn = ttk.Button(button_frame, text="Submit", command=on_submit)
    submit_btn.pack(side=tk.LEFT, padx=5)
    
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    cancel_btn.pack(side=tk.LEFT, padx=5)
    
    windll.shcore.SetProcessDpiAwareness(1)
    dialogue.grab_set()
    dialogue.wait_window()
    
    return result['selection']


    
