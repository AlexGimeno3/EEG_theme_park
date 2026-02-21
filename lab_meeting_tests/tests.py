from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from eeg_theme_park.utils.eeg_signal import EEGSignal
from eeg_theme_park.utils.signal_functions import add_power, add_artefact, art_reject
from eeg_theme_park.modes.playground.file_commands import build_signal_object, save_signal
import numpy as np
import copy
import datetime as dt


"""
Some useful math notes.
Any signal of frequency 2pif with initial amplitude Ai and final amplitude Af, increasing at a linear rate across the total signal time ttot and assuming a phase of 0, can be expressed as Ai*sin(2*pi*f*t) + m*t*sin(2*pi*f*t). Therefore, to build such a wave, we initialize the constant portion of the signal, then create the varying portion.

"""

def build_tests():
    save_dir = Path(r"C:\Users\Alex_G\Proton Drive\agimeno310\My files\Personal Projects\Autodidact\eeg_theme_park\lab_meeting_tests\test_signals_new")
    #Battery 1: verify our functions and extracted statistics work on ideal data.
    #Test 1.1: verify the 8to10Hz function. Specifically, build a one-hour, 9 Hz signal with amplitude 10. Power is the time-averaged mean-squared value of the signal; for a sin wave, this comes out to A^2/2 (this can be derived). Therefore, power should be 50 uV^2.
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
    print("Saved test 1.1")
    #Test 1.2: verify  One-hour signal starting at 9 Hz at 10 uV, increasing to 9 Hz at 20 uV, phase = 0. TSS of 8to10hz power should be (400/2-100/2)/3600 = 0.04167
    t1_2_signal_specs = {
        "name" : "test_1_2",
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
    print("Saved test 1.2")

    

    #Battery 2: assess our functions and extracted statistics on synthetic noisy data with artefacts rejected.
    #Test 2.1: same as test 1.1 but with 3 broadband noise artefacts making up 10% of the signal
    t2_1_signal_specs = {
        "name" : "test_2_1",
        "amp" : 10,
        "freq" : 9,
        "phase":0,
        "srate": 250,
        "time_length":3600
    }
    signal_t2_1 = build_signal_object(signal_specs = t2_1_signal_specs)
    t2_1_path = save_dir/f"{signal_t2_1.name}.pkl"
    #Make 10% of signal broadband electrical artefact
    noiser = add_artefact(num_artefacts= 3, proportion_artefacts = 0.10)
    signal_t2_1 = noiser.apply(signal_t2_1)
    save_signal(eeg_signal_obj = signal_t2_1, file_path = t2_1_path)
    print("Saved test 2.1")
    #Test 2.2: same as test 1.2 but with 3 broadband noise artefacts making up 10% of the signal
    t2_2_signal_specs = {
        "name" : "test_2_2",
        "amp" : 10,
        "freq" : 9,
        "phase":0,
        "srate": 250,
        "time_length":3600
    }
    signal_t2_2 = build_signal_object(signal_specs = t2_2_signal_specs)
    t2_2_path = save_dir/f"{signal_t2_2.name}.pkl"
    power_adder = add_power(frequency = 9, amplitude = 0, final_amplitude = 10)
    signal_t2_2 = power_adder.apply(signal_t2_2)
    noiser = add_artefact(num_artefacts= 3, proportion_artefacts = 0.10)
    signal_t2_2 = noiser.apply(signal_t2_2)
    save_signal(eeg_signal_obj = signal_t2_2, file_path = t2_2_path)
    print("Saved test 2.2")

    #Battery 3: asses the code's ability to extract features after artefact rejection
    #Test 3.1: same as test 2.1 but with artefacts (10% of signal) removed
    signal_t3_1 = copy.deepcopy(signal_t2_1) #Copies so the artefact being removed is consistent across the two signals
    signal_t3_1.name = "test_3_1"
    t3_1_path = save_dir/f"{signal_t3_1.name}.pkl"
    #Make 10% of signal broadband electrical artefact
    cleaner = art_reject()
    signal_t3_1 = cleaner.apply(signal_t3_1)
    save_signal(eeg_signal_obj = signal_t3_1, file_path = t3_1_path)
    print("Saved test 3.1")
    #Test 3.2: same as test 2.2 but with artefacts (10% of signal) removed
    signal_t3_2 = copy.deepcopy(signal_t2_2) #Copies so the artefact being removed is consistent across the two signals
    signal_t3_2.name = "test_3_2"
    t3_2_path = save_dir/f"{signal_t3_2.name}.pkl"
    #Make 10% of signal broadband electrical artefact
    cleaner = art_reject()
    signal_t3_1 = cleaner.apply(signal_t3_2)
    save_signal(eeg_signal_obj = signal_t3_2, file_path = t3_2_path)
    print("Saved test 3.2")
    
    #Battery 4: asses the code's ability to correctly identify intraoperative time periods.
    #Test 4.1: assuming a 10 uV, 9 Hz signal is collected at 7 AM and has a transient increase in power to 20 uV from 7:30 to 7:35 AM. Power during this time should be 200 uV^2, 50 uV^2 elsewhere.
    power_start_time = 1800 #30 mins after start; i.e., 073000
    power_end_time = 2100 #35 mins after start; i.e., 073500
    t4_1_start_datetime = dt.datetime(2025, 12, 21, 7, 0, 0) #7 AM, Dec 21, 2025. The signal goes from 070000 to 080000
    t4_1_signal_specs = {
        "name" : "test_4_1",
        "amp" : 10,
        "freq" : 9,
        "phase":0,
        "srate": 250,
        "time_length":3600,
        "datetime_collected":t4_1_start_datetime
    }
    signal_t4_1 = build_signal_object(signal_specs = t4_1_signal_specs)
    t4_1_path = save_dir/f"{signal_t4_1.name}.pkl"
    power_adder = add_power(frequency = 9, amplitude = 10) #Power in the altered segment should be ((10+10)**2)/2 = 200 uV^2
    signal_t4_1 = power_adder.apply(signal_t4_1, time_range = [power_start_time, power_end_time])
    save_signal(eeg_signal_obj = signal_t4_1, file_path = t4_1_path)
    print("Saved test 4.1")
    #Test 4.2: verify 5000 to 6000 sec time period on test 1.2 signal
    t4_2_start_datetime = dt.datetime(2025, 12, 21, 13, 0, 0) #1 PM, Dec 21, 2025. The signal goes from 130000 to 140000
    power_start_time = 300 #Starts at 1:05 PM
    power_end_time = 2700 #Ends at 1:45 PM
    t4_2_signal_specs = {
        "name" : "test_4_2",
        "amp" : 10,
        "freq" : 9,
        "phase":0,
        "srate": 250,
        "time_length":3600,
        "datetime_collected":t4_2_start_datetime
    }
    signal_t4_2 = build_signal_object(signal_specs = t4_2_signal_specs)
    t4_2_path = save_dir/f"{signal_t4_2.name}.pkl"
    power_adder_original = add_power(frequency = 9, amplitude = 0, final_amplitude = 10)
    signal_t4_2 = power_adder_original.apply(signal_t4_2)
    power_adder_2 = add_power(frequency = 9, amplitude = 10) #Power in the altered segment should be > ((10+10)**2)/2 therefore > 200 uV^2
    signal_t4_2 = power_adder_2.apply(signal_t4_2, time_range = [power_start_time, power_end_time])
    save_signal(eeg_signal_obj = signal_t4_2, file_path = t4_2_path)
    print("Saved test 4.2")


if __name__ == "__main__":
    build_tests()