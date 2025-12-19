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