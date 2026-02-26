"""
EEG Visualizer classes. These produce visualizations of a signal without
modifying it or extracting time series features.
"""

from abc import ABC, abstractmethod
import numpy as np
import inspect
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import matplotlib.pyplot as plt
from eeg_theme_park.utils.gui_utilities import simple_dialogue


class EEGVisualizer(ABC):
    """
    Abstract base class for EEG visualizations.
    Visualizers do not modify the signal or create TimeSeries.
    """

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def params_units_dict(self):
        pass

    def __init__(self, **kwargs):
        self.args_dict = kwargs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if isinstance(cls.__dict__.get("name"), str):
            AllVisualizers.add_to_visualizers(cls)

    def apply(self, eeg_object, time_range=None):
        """
        Resolve time range then delegate to _visualize().
        Does not modify eeg_object.

        Inputs:
        - eeg_object (EEGSignal): signal to visualize
        - time_range (list or None): [start, end] in seconds. If None,
          falls back to analyze_time_lims, then full signal.

        Returns:
        - eeg_object (EEGSignal): unchanged
        """
        if time_range is None:
            if len(eeg_object.analyze_time_lims) > 0:
                time_range = [eeg_object.analyze_time_lims[0],
                              eeg_object.analyze_time_lims[1]]
            else:
                time_range = [eeg_object.times[0], eeg_object.times[-1]]

        start_idx = eeg_object.time_to_index(time_range[0])
        end_idx = eeg_object.time_to_index(time_range[1])

        self._visualize(eeg_object, start_idx, end_idx, time_range)
        return eeg_object

    @abstractmethod
    def _visualize(self, eeg_object, start_idx, end_idx, time_range):
        """
        Produce the visualization.

        Inputs:
        - eeg_object (EEGSignal): signal (read-only)
        - start_idx (int): start sample index
        - end_idx (int): end sample index (inclusive)
        - time_range (list): [start_sec, end_sec]
        """
        pass

    @classmethod
    def get_params(cls, eeg_object, parent=None):
        """
        Default get_params — mirrors EEGFunction.get_params using inspect.
        Subclasses may override for custom GUI (e.g., channel dropdowns).
        """
        sig = inspect.signature(cls.__init__)
        params_to_collect = {
            name: param for name, param in sig.parameters.items()
            if name not in ('self', 'kwargs', '__class__')
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY)
        }
        if len(params_to_collect) == 0:
            return {}

        ret_var = {}
        string_vars = {}

        def check_fields(*args):
            all_filled = True
            for pname, param in params_to_collect.items():
                if param.default is inspect.Parameter.empty:
                    if string_vars[pname].get().strip() == "":
                        all_filled = False
                        break
            submit_button.config(state='normal' if all_filled else 'disabled')

        def submit():
            nonlocal ret_var
            collected = {}
            for pname, str_var in string_vars.items():
                val = str_var.get().strip()
                if val == "":
                    p = params_to_collect[pname]
                    if p.default is inspect.Parameter.empty:
                        simple_dialogue(f"Parameter '{pname}' is required.")
                        return
                    collected[pname] = p.default
                elif val.lower() == "none":
                    collected[pname] = None
                else:
                    try:
                        collected[pname] = int(val)
                    except ValueError:
                        try:
                            collected[pname] = float(val)
                        except ValueError:
                            collected[pname] = val
            ret_var = collected
            dialogue.destroy()

        dialogue = tk.Toplevel(parent)
        dialogue.title(f"Parameters for {cls.name}")
        dialogue.geometry("600x400")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        ttk.Label(dialogue, text=f"Please enter parameters for '{cls.name}'",
                  wraplength=580).grid(column=0, row=0, columnspan=2, padx=10, pady=10)

        row = 1
        units_dict = cls.params_units_dict
        for pname, param in params_to_collect.items():
            label_text = pname.replace('_', ' ').title()
            if pname in units_dict:
                label_text += f" ({units_dict[pname]})"
            ttk.Label(dialogue, text=f"{label_text}:").grid(
                column=0, row=row, sticky="w", padx=10, pady=5)
            str_var = tk.StringVar(dialogue)
            if param.default is not inspect.Parameter.empty:
                str_var.set(str(param.default))
            str_var.trace_add('write', check_fields)
            ttk.Entry(dialogue, textvariable=str_var, width=30).grid(
                column=1, row=row, padx=10, pady=5)
            string_vars[pname] = str_var
            row += 1

        all_filled = all(
            params_to_collect[p].default is not inspect.Parameter.empty
            for p in params_to_collect
        )
        submit_button = ttk.Button(dialogue, text="Submit", command=submit,
                                   state='normal' if all_filled else 'disabled')
        submit_button.grid(column=0, columnspan=2, row=row, pady=20)
        check_fields()
        dialogue.wait_window()
        return ret_var if ret_var else None


class AllVisualizers:
    """Registry for all EEGVisualizer subclasses."""
    _visualizers = []

    @classmethod
    def add_to_visualizers(cls, Visualizer):
        cls._visualizers.append(Visualizer)

    @classmethod
    def get_visualizer_names(cls):
        return [v.name for v in cls._visualizers]


# ---------------------------------------------------------------------------
# Concrete visualizer: Power Spectral Density
# ---------------------------------------------------------------------------

class ViewPSD(EEGVisualizer):
    name = "View PSD"
    params_units_dict = {
        "save_path": "directory or None",
        "ext": "",
        "dpi": "dots per inch",
        "figsize": "inches (w, h)",
        "display": "0 or 1",
    }

    def __init__(self, channel=None, save_path=None, ext=".png", dpi=300,
                 figsize=(12, 4), display=0, **kwargs):
        """
        Inputs:
        - channel (str or None): channel to plot. None = current channel.
        - save_path (str or None): directory to save figure. None = don't save.
        - ext (str): file extension
        - dpi (int): resolution
        - figsize (tuple): figure size (w, h) in inches
        - display (int): 1 = show plot interactively, 0 = don't
        """
        params = {k: v for k, v in locals().items()
                  if k not in ('self', 'kwargs', '__class__')}
        super().__init__(**params, **kwargs)
        self.__dict__.update(params)

        if isinstance(self.figsize, str):
            self.figsize = tuple(
                float(x.strip()) for x in self.figsize.strip("()").split(","))
        self.dpi = int(self.dpi)
        self.display = int(self.display)

    @classmethod
    def get_params(cls, eeg_object, parent=None):
        """
        Custom get_params with a channel dropdown, a display checkbox,
        and text entries for the remaining parameters.
        """
        sig = inspect.signature(cls.__init__)
        # Exclude channel and display from text entries — handled separately
        params_to_collect = {
            name: param for name, param in sig.parameters.items()
            if name not in ('self', 'kwargs', '__class__', 'channel', 'display')
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY)
        }

        ret_var = {}
        string_vars = {}
        channel_var = None
        display_var = None

        def check_fields(*args):
            all_filled = True
            for pname, param in params_to_collect.items():
                if param.default is inspect.Parameter.empty:
                    if string_vars[pname].get().strip() == "":
                        all_filled = False
                        break
            submit_button.config(state='normal' if all_filled else 'disabled')

        def submit():
            nonlocal ret_var
            collected = {}
            for pname, str_var in string_vars.items():
                val = str_var.get().strip()
                if val == "":
                    p = params_to_collect[pname]
                    if p.default is inspect.Parameter.empty:
                        simple_dialogue(f"Parameter '{pname}' is required.")
                        return
                    collected[pname] = p.default
                elif val.lower() == "none":
                    collected[pname] = None
                else:
                    try:
                        collected[pname] = int(val)
                    except ValueError:
                        try:
                            collected[pname] = float(val)
                        except ValueError:
                            collected[pname] = val
            collected["channel"] = channel_var.get() if channel_var else None
            collected["display"] = 1 if display_var.get() else 0
            ret_var = collected
            dialogue.destroy()

        # --- Build dialogue ---
        dialogue = tk.Toplevel(parent)
        dialogue.title(f"Parameters for {cls.name}")
        dialogue.geometry("600x500")
        dialogue.grab_set()
        dialogue.lift()
        dialogue.focus_force()

        ttk.Label(dialogue, text=f"Please enter parameters for '{cls.name}'",
                  wraplength=580).grid(column=0, row=0, columnspan=2, padx=10, pady=10)

        row = 1

        # Channel dropdown
        ttk.Label(dialogue, text="Channel:").grid(
            column=0, row=row, sticky="w", padx=10, pady=5)
        channel_var = tk.StringVar(dialogue)
        if hasattr(eeg_object, 'all_channel_labels') and len(eeg_object.all_channel_labels) > 0:
            channel_var.set(eeg_object.current_channel)
            channel_widget = ttk.Combobox(
                dialogue, textvariable=channel_var,
                values=list(eeg_object.all_channel_labels),
                state="readonly", width=27)
        else:
            channel_widget = ttk.Entry(dialogue, textvariable=channel_var, width=30)
        channel_widget.grid(column=1, row=row, padx=10, pady=5)
        row += 1

        # Display checkbox
        ttk.Label(dialogue, text="Display plot:").grid(
            column=0, row=row, sticky="w", padx=10, pady=5)
        display_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialogue, variable=display_var).grid(
            column=1, row=row, sticky="w", padx=10, pady=5)
        row += 1

        # Text entries for remaining params
        units_dict = cls.params_units_dict
        for pname, param in params_to_collect.items():
            label_text = pname.replace('_', ' ').title()
            if pname in units_dict:
                label_text += f" ({units_dict[pname]})"
            ttk.Label(dialogue, text=f"{label_text}:").grid(
                column=0, row=row, sticky="w", padx=10, pady=5)
            str_var = tk.StringVar(dialogue)
            if param.default is not inspect.Parameter.empty:
                str_var.set(str(param.default))
            str_var.trace_add('write', check_fields)
            ttk.Entry(dialogue, textvariable=str_var, width=30).grid(
                column=1, row=row, padx=10, pady=5)
            string_vars[pname] = str_var
            row += 1

        all_filled = all(
            params_to_collect[p].default is not inspect.Parameter.empty
            for p in params_to_collect
        )
        submit_button = ttk.Button(dialogue, text="Submit", command=submit,
                                   state='normal' if all_filled else 'disabled')
        submit_button.grid(column=0, columnspan=2, row=row, pady=20)
        check_fields()
        dialogue.wait_window()
        return ret_var if ret_var else None

    def _visualize(self, eeg_object, start_idx, end_idx, time_range):
        """Compute and plot power spectral density using Welch's method."""
        from scipy.signal import welch

        ch = self.channel if self.channel else eeg_object.current_channel
        if ch not in eeg_object.all_channel_data:
            simple_dialogue(
                f"Channel '{ch}' not found. "
                f"Available: {eeg_object.all_channel_labels}")
            return

        data = eeg_object.all_channel_data[ch][start_idx:end_idx + 1]
        srate = eeg_object.srate

        # NaN-tolerant: zero out NaNs so FFT doesn't propagate them
        data_clean = np.where(np.isnan(data), 0, data)

        nperseg = min(int(srate * 2), len(data_clean))
        if nperseg < 4:
            simple_dialogue(
                "Not enough data in the selected range to compute a PSD.")
            return

        noverlap = nperseg // 2
        freqs, pxx = welch(data_clean, fs=srate,
                           nperseg=nperseg, noverlap=noverlap)

        # Convert to dB
        pxx_db = 10 * np.log10(pxx + 1e-12)

        # Cap at a reasonable EEG frequency
        max_freq = min(srate / 2, 80)
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        pxx_db = pxx_db[freq_mask]

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(freqs, pxx_db, linewidth=0.8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.set_title(
            f'PSD: {eeg_object.name} — Channel: {ch} '
            f'({time_range[0]:.1f}–{time_range[1]:.1f} s)')
        ax.grid(True)

        if self.save_path is not None:
            save_full = (Path(self.save_path)
                         / f"{eeg_object.name}_{ch}_psd{self.ext}")
            fig.savefig(save_full, dpi=self.dpi, bbox_inches='tight')
            print(f"PSD saved to {save_full}")

        if self.display == 1:
            plt.show()
        else:
            plt.close(fig)