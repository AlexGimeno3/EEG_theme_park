from eeg_theme_park.utils.eeg_signal import EEGSignal
from eeg_teme_park.utils.signal_functions import add_power
from eeg_theme_park.modes.playground.file_commands import build_signal_object, save_signal
import numpy as np
from pathlib import Path


"""
Some useful math notes.
Any signal of frequency 2pif with initial amplitude Ai and final amplitude Af, increasing at a linear rate across the total signal time ttot and assuming a phase of 0, can be expressed as Ai*sin(2*pi*f*t) + m*t*sin(2*pi*f*t). Therefore, to build such a wave, we initialize the constant portion of the signal, then create the varying portion.

"""

def build_tests():
    save_dir = Path(r"C:\Users\Alex_G\Proton Drive\agimeno310\My files\Personal Projects\Autodidact\eeg_theme_park\tests\test_signals")
    #Battery 1: verify our functions and extracted statistics work on ideal data.
    #Test 1.1: verify the 8to10Hz function. Specifically, build a one-hour, 9 Hz signal with amplitude 10. Median power should be 100.
    t1_1_signal_specs = {
        "name" : "test_1_1",
        "amp" : 10,
        "freq" : 9,
        "phase":0,
        "srate": 250,
        "time_length":3600
    }
    signal_t1_1 = build_signal_object(signal_specs = t1_1_signal_specs)
    t1_1_path = save_dir/f"{signal_t1_1.name}.pkl"
    save_signal(eeg_signal_obj = signal_t1_1, file_path = t1_1_path)
    
    #Test 1.2: verify  One-hour signal starting at 9 Hz at 10 uV, increasing to 9 Hz at 20 uV, phase = 0. TSS of 8to10hz power should be 10/3600 = 0.00278
    t1_2_signal_specs = {
        "name" : "test_1_1",
        "amp" : 10,
        "freq" : 9,
        "phase":0,
        "srate": 250,
        "time_length":3600
    }
    signal_t1_2 = build_signal_object(signal_specs = t1_2_signal_specs)
    t1_2_path = save_dir/f"{signal_t1_2.name}.pkl"
    power_adder = add_power(frequency = 9, amplitude = 0, final_amplitude = 10)
    signal_t1_2 = power_adder.apply(signal_t1_2)
    save_signal(eeg_signal_obj = signal_t1_2, file_path = t1_2_path)

    

    #Battery 2: assess our functions and extracted statistics on synthetic noisy data with artefacts rejected.
    #Test 2.1
    #Test 2.2

    #Battery 3: asses the code's ability to correctly identify intraoperative time periods


if __name__ == "__main__":
    build_tests()