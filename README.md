**DISCLAIMER: EEG Theme Park is still in development! It still needs testing and validation on synthetic and real data before it is ready for research use. It should NEVER be used for clinical applications.**

EEG Theme Park is a GUI-centered EEG analysis software whose goal is to make quantitative EEG analysis accessible to those with little coding experience.

How it Works
The main workhorses of EEG Theme Park are Analyzers and Signal Functions. Analyzers are simply functions that extract characteristics of an EEG signal (e.g., α power). While implementation of NEURALPY toolbox functions will be coming soon, more tech-savvy users may also implement their own functions. EEG Theme Park is built so that these Analyzers act on small segments of a longer signal, creating a new time series the user to visualize and extract data about how the characteristic the Analyzer extracts changes over time (i.e., by extracting the slope of the new time series). Summary statistics (namely median, IQR, Theil-Sen slope, and Mann-Kendall τ) can be displayed in an Excel sheet later on. Signal Functions, on the other hand, alter a signal in some way; these may include filters or artefact rejection functions. Users may string these functions together into Pipelines, which specify the order in which to apply Signal Functions and Analyzers to an EEG file. In the software, users can then store all their EEG files in one directory and run a Pipeline on all files, resulting in an Excel sheet containing values for all summary statistics for all Analyzers used. Users may also export Pipelines, which can be sent to other users and uploaded into EEG Theme Park, enabling exact replication of an experimental protocol across mutliple users. EEG Theme Park also supports intraoperative time point specification; therefore, if users are interested in only a certain time period of a recording (e.g., from 1115 to 1315, perhaps during cardiopulmonary bypass), they may specify this in an auxiliary Excel file. This functionality currently only works for recordings that occur all during the same calendar day.

EEG Theme Park also includes a playground mode, which allows users to easily create, alter, and apply analyzers to their own synthetic signals, allowing them to easily see how a certain EEG processing technique behaves with different types of data.

EEG Theme park has also been built to work programatically for users who would rather not use the GUI features.






File Structure:

eeg_theme_park/                 # Project root
├── __main__.py                 # Entry point
├── __init__.py                 # Init
├── utils/                      # Shared utilities across all modes
│   ├── __init__.py
│   ├── eeg_analyzers.py        # EEGAnalyzer abstract class and AllAnalyzers class (for extracting parameters)
│   ├── eeg_signal.py           # EEGSignal and TimeSeries classes
│   ├── gui_utilities.py        # Shared GUI utilities
│   ├── loaders.py              # Data loaders (shared)
│   └── signal_functions.py     # EEGFunction abstract class and AllFunctions class  (for altering signal)
├── modes/                      # All application modes
│   ├── __init__.py
│   ├── mode_manager.py         # Mode management system
│   ├── playground/             # Signal generation/playground mode
│   │   ├── __init__.py
│   │   ├── playground.py               # PlaygroundMode class
│   │   └── file_commands.py            # Commands for the file menu
│   └── batch_processing/       # Batch processing mode
│       ├── __init__.py
│       ├── batch_processing.py               # BatchProcessingMode class
│       └── batch_commands.py
└── signals_cache/              # Data storage
