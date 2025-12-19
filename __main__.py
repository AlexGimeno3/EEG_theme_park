import tkinter as tk
from eeg_theme_park.modes.mode_manager import ModeManager
from eeg_theme_park.modes.playground.playground import PlaygroundMode
from eeg_theme_park.modes.batch_processing.batch_processing import BatchProcessing


def main():
    """Entry point for the application"""
    root = tk.Tk()
    root.title("EEG Theme Park")
    
    # Set up window size
    user_screen_width = root.winfo_screenwidth()
    user_screen_height = root.winfo_screenheight()
    root.geometry(f"{int(user_screen_width-100)}x{int(user_screen_height-100)}+10+45")
    
    try:
        manager = ModeManager(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print("=" * 60)
        print("ERROR STARTING APPLICATION:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()