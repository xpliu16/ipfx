import numpy as np
import logging
import pandas as pd
import scipy.fftpack as fftpack
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from collections import defaultdict

from ipfx.error import FeatureError
from ipfx.subthresh_features import baseline_voltage
from . import feature_vectors as fv
from . import time_series_utils as tsu

CHIRP_CODES = [
            "C2CHIRP180503", # current version, single length
            "C2CHIRP171129", # early version, three lengths
            "C2CHIRP171103", # only one example
        ]


def extract_chirp_features_by_sweep(sweepset, **params):
    results = []
    for sweep in sweepset.sweeps:
        try:
            amp, phase, freq = chirp_sweep_amp_phase(sweep, **params)
            result = chirp_sweep_features(amp, freq, phase)
            result['sweep_number'] = sweep.sweep_number
            results.append(result)
        except FeatureError as exc:
            logging.info(exc)
        except Exception:
            msg = F"Error processing chirp sweep {sweep.sweep_number}."
            logging.warning(msg, exc_info=True)

    if len(results)==0:
        logging.warning("No chirp sweep results available.")
        return {}

    mean_results = {key: np.mean([res[key] for res in results]) for key in results[0]}
    mean_results['sweeps'] = results
    return mean_results

def extract_chirp_fft(sweepset, min_freq=0.4, max_freq=40.0, **params):
    amps = []
    phases = []
    freqs = []
    for i, sweep in enumerate(sweepset.sweeps):
        try:
            amp, phase, freq = chirp_sweep_amp_phase(sweep, min_freq=min_freq, max_freq=max_freq, **params)
            amps.append(amp)
            phases.append(phase)
            freqs.append(freq)
        except (FeatureError, ValueError) as exc:
            logging.warning(exc)
    if len(amps)==0:
        raise FeatureError('No valid chirp sweeps available.')
    # in case of multiple distinct data shapes
    sizes = np.array([len(amp) for amp in amps])
    if not all([s==sizes[0] for s in sizes]):
        logging.warning(f'Unequal length chirp transforms: {sizes}.')
        unique, counts = np.unique(sizes, return_counts=True)
        common_size = unique[np.argmax(counts)]
        common_ind = np.where(sizes==common_size)
        amps = np.array(amps)[common_ind]
        phases = np.array(phases)[common_ind]
        freqs = np.array(freqs)[common_ind]
        
    amp = np.stack(amps).mean(axis=0)
    phase = np.stack(phases).mean(axis=0)
    low_freq_max = min_freq + 0.1
    results = chirp_sweep_features(amp, freq, phase, low_freq_max=low_freq_max)
    return results

def extract_chirp_peaks(sweepset, **params):
    amps = []
    min_freq=None
    max_freq=None
    for i, sweep in enumerate(sweepset.sweeps):
        try:
            amp, freq = amp_response_asymmetric(sweep, min_freq=min_freq, max_freq=max_freq, **params)
            amps.append(amp)
            # apply the frequency bounds from the first sweep to the others
            if min_freq is None:
                min_freq = freq[0]
                max_freq = freq[-1]
        except FeatureError as exc:
            logging.warning(exc)
    if len(amps)==0:
        raise FeatureError('No valid chirp sweeps available.')
    amp = np.stack(amps).mean(axis=0)
    low_freq_max = min_freq + 0.1
    results = chirp_sweep_features(amp, freq, low_freq_max=low_freq_max)
    return results

def amp_response_asymmetric(sweep, min_freq=None, max_freq=None, n_freq=500, freq_sigma=0.25):
    width = 8
    # TODO: v0 from baseline vs mean?
    sweep.align_to_start_of_epoch('stim')
    sweep.select_epoch('experiment')
    # v0 = baseline_voltage(sweep.t, sweep.v, start=0, baseline_interval=0.025)
    v0 = np.mean(sweep.v)
    sweep.select_epoch('stim')
    t = sweep.t

    v = tsu.subsample_average(sweep.v, width)
    i = tsu.subsample_average(sweep.i, width)

    fs = 1 / (sweep.t[1] * width)

    i_crossings = np.nonzero(np.diff(i>0))[0]
    i_peaks = np.array([np.argmax(np.abs(i[j:k])) + j for j, k in
                        zip(i_crossings[:-1], i_crossings[1:])])
    i_freq = fs/(2*np.diff(i_crossings))
    freq_fcn = interp1d(i_peaks, i_freq, assume_sorted=True, kind=2, )
    
    v = (v-v0)
    v_crossings = np.nonzero(np.diff(v>0))[0]
#     filter out noise crossings
    v_crossings = np.delete(v_crossings, np.nonzero(np.diff(v_crossings)<5))
    v_peaks = np.array([np.argmax(np.abs(v[j:k])) + j for j, k in
                        zip(v_crossings[:-1], v_crossings[1:])])
    
    v_peaks = v_peaks[(i_peaks[0] <= v_peaks) & (i_peaks[-1] >= v_peaks)]
    v_freq = freq_fcn(v_peaks)
    amp = np.abs(v[v_peaks])/np.max(i)
    upper = (v[v_peaks]>0)
    lower = (v[v_peaks]<0)
    
    min_freq = min_freq or i_freq[0]
    max_freq = max_freq or i_freq[-1]
    if i_freq[0]-min_freq > freq_sigma or max_freq-i_freq[-1] > freq_sigma:
        raise FeatureError(
            f"Chirp sweep{sweep.sweep_number} doesn't span desired frequencies.")
    freq = np.linspace(min_freq, max_freq, n_freq)
    amp_upper = gauss_smooth(v_freq[upper], amp[upper], freq, freq_sigma)
    amp_lower = gauss_smooth(v_freq[lower], amp[lower], freq, freq_sigma)
    amp = np.stack([amp_upper, amp_lower]).mean(axis=0)
    return amp, freq

def chirp_sweep_amp_phase(sweep, min_freq=0.4, max_freq=40.0, filter_bw=1, filter=True, **transform_params):
    """ Calculate amplitude and phase of chirp response

    Parameters
    ----------
    sweep_set: Sweep
        Set of chirp sweeps
    min_freq: float (optional, default 0.4)
        Minimum frequency for output to contain
    max_freq: float (optional, default 40)
        Maximum frequency for output to contain

    Returns
    -------
    amplitude: array
        Aka resistance
    phase: array
        Aka reactance
    freq: array
        Frequencies for amplitude and phase results
    """
    v, i, freq = transform_sweep(sweep, **transform_params)
    Z = v / i
    amp = np.abs(Z)
    phase = np.angle(Z)
    
    # window before or after smoothing?
    low_ind = tsu.find_time_index(freq, min_freq)
    high_ind = tsu.find_time_index(freq, max_freq)
    amp, phase, freq = map(lambda x: x[low_ind:high_ind], [amp, phase, freq])
    
    if filter:
        # pick odd number, approx number of points for smooth_bw interval
        n_filt = int(np.rint(filter_bw/2/(freq[1]-freq[0])))*2 + 1
        filt = lambda x: savgol_filter(x, n_filt, polyorder=1)
        amp, phase = map(filt, [amp, phase])

    return amp, phase, freq

def transform_sweep(sweep, n_sample=10000):
    """ Calculate Fourier transform of sweep current and voltage
    """
    sweep.select_epoch("stim")
    if np.all(sweep.v[-10:] == 0):
        raise FeatureError("Chirp stim epoch truncated.")
    v = sweep.v
    i = sweep.i
    t = sweep.t
    N = len(v)

    width = int(N / n_sample)
    pad = int(width*np.ceil(N/width) - N)
    v = tsu.subsample_average(v, width)
    i = tsu.subsample_average(i, width)
    dt = t[width] - t[0]

    nfreq = len(v)//2
    freq = np.linspace(0.0, 1.0/(2.0*dt), nfreq)

    v_fft = fftpack.fft(v)
    i_fft = fftpack.fft(i)

    return v_fft[:nfreq], i_fft[:nfreq], freq

def chirp_sweep_features(amp, freq, phase=None, low_freq_max=1.0):
    """Calculate a set of characteristic features from the impedance amplitude profile (ZAP).
    Peak response is measured relative to a low-frequency average response.

    Args:
        amp (ndarray): impedance amplitude (generalized resistance)
        freq (ndarray): frequencies corresponding to amp responses
        phase (ndarray, optional): phase shifts of response if not provided, phase features ignored.
        low_freq_max (float, optional): Upper frequency cutoff for low-frequency average reference value. Defaults to 1.5.

    Returns:
        dict: features
    """    
    i_max = np.argmax(amp)
    z_max = amp[i_max]
    i_3db = np.flatnonzero(amp > z_max/np.sqrt(2))[-1]
    low_freq_amp = np.mean(amp[freq < low_freq_max])
    features = {
        "peak_ratio": z_max/low_freq_amp,
        "peak_freq": freq[i_max],
        "3db_freq": freq[i_3db],
        "peak_impedance": z_max,
        "low_freq_impedance": low_freq_amp,
    }
    if phase is not None:
        if any(phase > 0):
            i_sync = np.flatnonzero(phase > 0)[-1]
            dfreq = freq[1] - freq[0]
            total_phase = np.mean(phase[phase>0])*dfreq
        else:
            i_sync = 0
            total_phase = 0
        phase_features = {
            "sync_freq": freq[i_sync],
            "phase_peak": phase[i_max],
            "phase_low": phase[0],
            "total_inductive_phase": total_phase,
        }
        features.update(phase_features)
    return features

def gauss_smooth(x, y, x_eval, sigma):
    """Apply smoothing to irregularly sampled data using a gaussian kernel

    Args:
        x (ndarray): 1D array of points y is sampled at
        y (ndarray): 1D array of data to be smoothed
        x_eval (ndarray): 1D array of points to evaluate smoothed function at
        sigma (float): standard deviation of gaussian kernel

    Returns:
        ndarray: 1D array of smoothed data evaluated at x_eval
    """    
    delta_x = x_eval[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, y)
    return y_eval
