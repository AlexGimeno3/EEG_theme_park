from eeg_theme_park.utils.eeg_analyzers import EEGAnalyzer
from eeg_theme_park.utils.signal_functions import EEGFunction
import pandas as pd
import copy
from scipy.stats import theilslopes, kendalltau
import numpy as np
import gc

"""
File containing the Pipeline class, which is used to create exportable and loadable EEG processing pipelines between users.
"""

def find_clean_segments(data, srate, min_length_sec):
    """
    Find all contiguous non-NaN segments in data that are >= min_length_sec.
    
    Inputs:
    - data (numpy array): signal data that may contain NaN values
    - srate (float): sampling rate in Hz
    - min_length_sec (float): minimum segment length in seconds
    
    Outputs:
    - clean_segments (list of tuples): [(start_idx, end_idx), ...] where each tuple
      represents a clean segment with end_idx being exclusive (Python slice convention)
    """
    is_valid = ~np.isnan(data)
    
    # Find transitions between valid and invalid data
    # Pad with False to catch segments at start/end
    transitions = np.diff(np.concatenate(([False], is_valid, [False])).astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    min_samples = int(min_length_sec * srate)
    
    clean_segments = []
    for start, end in zip(starts, ends):
        if (end - start) >= min_samples:
            clean_segments.append((start, end))
    
    return clean_segments

class Pipeline:
    """
    Pipeline class. This is a class that contains all the steps to take data from raw EEG files to a final Excel.
    """
    analyzer_statistics = [
        {'key': 'mean', 'suffix': 'mean', 'unit_transform': lambda u: u},
        {'key': 'median', 'suffix': 'median', 'unit_transform': lambda u: u},
        {'key': 'iqr', 'suffix': 'IQR', 'unit_transform': lambda u: u},
        {'key': 'theil_sen_slope', 'suffix': 'TSS', 'unit_transform': lambda u: f"{u}/sec" if u else "per sec"},
        {'key': 'mann_kendall_tau', 'suffix': 'MKT', 'unit_transform': lambda u: ""}  # No units for MKT
    ]
    
    def __init__(self, operations, min_clean_length = 0):
        """
        Inputs:
        - operations (arr of EEGFunction or EEGAnalyzer subclasses): sequantial array of operations that will be applied to code one-by-one
        """
        self.operations = operations
        self.min_clean_length = min_clean_length
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
            functions = [op for op in self.operations if isinstance(op, EEGFunction)]
            analyzers = [op for op in self.operations if isinstance(op, EEGAnalyzer)]

            # Step 1: Apply all EEGFunctions. NB: the apply subclass forces the minimum length with each consecutive function.
            for func in functions:
                eeg_signal = func.apply(eeg_signal, time_range=None, min_clean_length=self.min_clean_length)
                if not self.ever_run:
                    self.pipeline_log += f"\nApplied function {func.name} with specs {func.args_dict}"
            
            # Step 2: After all functions, find final clean segments PER CHANNEL and mark short ones as NaN
            clean_segments = None  # Will become a dict {ch_name: [(start, end), ...]}
            if self.min_clean_length > 0:
                clean_segments = {}
                original_channel = eeg_signal.current_channel

                for ch_name in eeg_signal.all_channel_labels:
                    ch_data = eeg_signal.all_channel_data[ch_name]
                    ch_segs = find_clean_segments(ch_data, eeg_signal.srate, self.min_clean_length)
                    clean_segments[ch_name] = ch_segs

                    # Mark all data NOT in clean segments as NaN for this channel
                    mask = np.ones(len(ch_data), dtype=bool)
                    for start, end in ch_segs:
                        mask[start:end] = False
                    eeg_signal.all_channel_data[ch_name][mask] = np.nan

                eeg_signal.current_channel = original_channel
                
                # Calculate statistics for logging (using primary channel as representative)
                primary_segs = clean_segments.get(eeg_signal.current_channel, [])
                total_samples = len(eeg_signal.data)
                clean_samples = sum(end - start for start, end in primary_segs)
                clean_pct = 100 * clean_samples / total_samples if total_samples > 0 else 0
                
                if not self.ever_run:
                    self.pipeline_log += (
                        f"\n\nClean segment analysis (min length = {self.min_clean_length}s):"
                        f"\n  - Found {len(clean_segments)} clean segments"
                        f"\n  - Clean data: {clean_pct:.1f}% of signal"
                        f"\n  - Marked segments shorter than {self.min_clean_length}s as NaN"
                    )
            
            # Step 3: Apply all EEGAnalyzers (pass clean_segments)
            results_arr = []
            for analyzer in analyzers:
                eeg_signal = analyzer.apply(eeg_signal, clean_segments=clean_segments)
                
                my_ts_data = np.array(eeg_signal.time_series[-1].values)
                my_ts_times = np.array(eeg_signal.time_series[-1].times)
                
                mask = ~np.isnan(my_ts_data)
                my_ts_data = my_ts_data[mask]
                my_ts_times = my_ts_times[mask]
                
                if len(my_ts_data) > 0:
                    mean_val = np.mean(my_ts_data)
                    median_val = np.median(my_ts_data)
                    q75, q25 = np.percentile(my_ts_data, [75, 25])
                    iqr_val = q75 - q25
                    tss_result = theilslopes(my_ts_data, my_ts_times)
                    TSS_val = tss_result[0]
                    mkt_result = kendalltau(my_ts_times, my_ts_data)
                    mkt_val = mkt_result.correlation
                else:
                    mean_val = np.nan
                    median_val = np.nan
                    iqr_val = np.nan
                    TSS_val = np.nan
                    mkt_val = np.nan
                
                result_dict = {
                    'analyzer_name': analyzer.name,
                    'mean': mean_val,
                    'median': median_val,
                    'iqr': iqr_val,
                    'theil_sen_slope': TSS_val,
                    'mann_kendall_tau': mkt_val
                }
                results_arr.append(result_dict)
                
                if not self.ever_run:
                    self.pipeline_log += f"\nApplied analyzer {analyzer.name} with specs {analyzer.args_dict}"
        
        except Exception as e:
            error_message = f"Error processing {id}: {str(e)}"
            print(error_message)
            return None, error_message
        
        finally:
            # Explicitly delete and garbage collect
            if 'eeg_signal' in locals():
                del eeg_signal
            gc.collect()  # Force garbage collection
        
        self.ever_run = True
        return results_arr, None
