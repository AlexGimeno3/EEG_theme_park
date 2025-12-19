from eeg_theme_park.utils.eeg_analyzers import EEGAnalyzer
from eeg_theme_park.utils.signal_functions import EEGFunction
import pandas as pd
import copy
from scipy.stats import theilslopes, kendalltau
import numpy as np

"""
File containing the Pipeline class, which is used to create exportable and loadable EEG processing pipelines between users.
"""
class Pipeline:
    """
    Pipeline class. This is a class that contains all the steps to take data from raw EEG files to a final Excel.
    """
    analyzer_statistics = [
        {'key': 'median', 'suffix': 'median', 'unit_transform': lambda u: u},
        {'key': 'iqr', 'suffix': 'IQR', 'unit_transform': lambda u: u},
        {'key': 'theil_sen_slope', 'suffix': 'TSS', 'unit_transform': lambda u: f"{u}/sec" if u else "per sec"},
        {'key': 'mann_kendall_tau', 'suffix': 'MKT', 'unit_transform': lambda u: ""}  # No units for MKT
    ]
    
    def __init__(self, operations):
        """
        Inputs:
        - operations (arr of EEGFunction or EEGAnalyzer subclasses): sequantial array of operations that will be applied to code one-by-one
        """
        self.operations = operations
        self.pipeline_log = ""
        self.ever_run = False


    def get_expected_columns(self):
        """
        Returns a list of column definitions that this pipeline will generate.
        Each definition is a dict with 'name' and 'full_name' (with units).
        """
        columns = [{'name': 'ID', 'full_name': 'ID'}]
        
        # Track duplicate analyzer names
        name_counts = {}
        
        for op in self.operations:
            if isinstance(op, EEGAnalyzer):
                base_name = op.name
                units = op.units
                
                # Handle duplicate names
                if base_name in name_counts:
                    name_counts[base_name] += 1
                    modified_name = f"{base_name}_{name_counts[base_name]}"
                else:
                    name_counts[base_name] = 0
                    modified_name = base_name
                
                # Create a column for each statistic
                for stat_def in self.analyzer_statistics:
                    col_base = f"{modified_name}_{stat_def['suffix']}"
                    transformed_units = stat_def['unit_transform'](units)
                    
                    if transformed_units:
                        col_full = f"{col_base} ({transformed_units})"
                    else:
                        col_full = col_base
                    
                    columns.append({
                        'name': col_base,
                        'full_name': col_full,
                        'analyzer_name': modified_name,
                        'statistic': stat_def['key']
                    })
        
        return columns


    
    def run_pipeline(self, eeg_signal):
        """
        Method that runs the entire pipeline, adding the name for each eeg_signal (as well as the result) to self.results_df.

        Inputs:
        - eeg_signal (EEGSignal subclass): the EEGSignal we would like to run our pipeline on
        
        Ouputs:
        - results_row (arr of dict): dicts with all outputs from the EEG analyzers. each dictionary reflects one analyzer, and contains the median, iqr, theil-sen-slope, and mann-kendall tau of each generated time series 
        - error_row (arr of str OR None): 2-item array with the ID and the error message (used for error logging). If no error, will be None
        """
        eeg_signal = copy.deepcopy(eeg_signal)
        id = eeg_signal.name

        try:
            results_arr = []
            for op in self.operations:
                if isinstance(op, EEGFunction): #Apply function to whole signal
                    eeg_signal = op.apply(eeg_signal, time_range=None)  # This applies the function to the whole signal
                    if not self.ever_run:
                        self.pipeline_log += f"\nApplied the function {op.name} with specs {op.args_dict}"
                elif isinstance(op, EEGAnalyzer): #Apply function to eeg_signal
                    eeg_signal = op.apply(eeg_signal)
                    my_ts_data = np.array(eeg_signal.time_series[-1].values)
                    my_ts_times = np.array(eeg_signal.time_series[-1].times)
                    
                    # Drop all NaN values in my_ts_data as well as their respective indices/time points in my_ts_times
                    mask = ~np.isnan(my_ts_data)
                    my_ts_data = my_ts_data[mask]
                    my_ts_times = my_ts_times[mask]
                    
                    # Calculate median
                    median_val = np.median(my_ts_data)
                    
                    # Calculate IQR (interquartile range)
                    q75, q25 = np.percentile(my_ts_data, [75, 25])
                    iqr_val = q75 - q25
                    
                    # Calculate Theil-Sen slope (robust linear regression slope)
                    # This accounts for non-equally spaced time points
                    tss_result = theilslopes(my_ts_data, my_ts_times)
                    TSS_val = tss_result[0]  # The slope estimate
                    
                    # Calculate Mann-Kendall tau value
                    # This is a measure of monotonic trend
                    mkt_result = kendalltau(my_ts_times, my_ts_data)
                    mkt_val = mkt_result.correlation  # Kendall's tau correlation coefficient
                    
                    # Store results (you can modify this structure as needed)
                    result_dict = {
                        'analyzer_name': op.name,
                        'median': median_val,
                        'iqr': iqr_val,
                        'theil_sen_slope': TSS_val,
                        'mann_kendall_tau': mkt_val
                    }
                    results_arr.append(result_dict)
                    if not self.ever_run:
                        self.pipeline_log += f"\nApplied the function {op.name} with specs {op.args_dict}"
                else:
                    raise ValueError(f"The operation in the pipeline was neither an EEGFunction subclass nor an EEGAnalyzer subclass. Double-check where you got your pipeline from.")
        except Exception as e:
            error_message = f"Error processing {id}: {str(e)}"
            print(error_message)
            return None, error_message
        
        # If successful, return results and no error
        del eeg_signal
        self.ever_run = True
        return results_arr, None
