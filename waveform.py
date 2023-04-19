import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
from sys import argv
from os.path import splitext, exists
from lmfit.models import LinearModel, GaussianModel
from scipy.signal import find_peaks

# Resolution of the CAEN digitiser
digi_res = 4 # ns
qhist_bins = 200

def fit_wform(wform):
    """
    Fits the waveform, assuming linear background and gaussian peak.

    :param np.array wform: Numpy array of the waveform to fit.
    """
    mod_bg = LinearModel(prefix="lin_")
    mod_peak = GaussianModel(prefix="g1_")

    model = mod_bg + mod_peak

    # Guess the center as the global minimum, scaled by digitiser res
    g1_center = digi_res*np.argmin(wform)

    params = model.make_params(g1_amplitude=-20, g1_center=g1_center, 
        g1_sigma=2, lin_amplitude=120)

    # Scale x to fit to real time values
    xs = [digi_res*x for x in range(len(wform))]
    result = model.fit(wform, params, x=xs)

    # print(result.fit_report())

    return result

def find_peaks(qs):
    """
    Finds peaks using scipy.signal.find_peaks, in the integrated charge
    histogram. Doesn't work very well.

    :param list of int qs: The integrated charges from each individual waveform.
    """
    qs_hist = np.histogram(qs, bins=qhist_bins)
    print(qs_hist)

    bin_width = qs_hist[1][1] - qs_hist[1][0]
    plt.bar(qs_hist[1][:-1],qs_hist[0],width=bin_width)
    plt.yscale("log")

    peaks = find_peaks(qs_hist[0])[0]
    print("%i peak(s) found with indices: " % len(peaks), end="")
    print(peaks)

    plt.vlines([peak*bin_width for peak in peaks],0,1e5)
    plt.show()

    return

def fit_qhist(qs):
    """
    Fits Gaussians to the integrated charge histogram, fitting the pedestal, 1pe
    and 2pe peaks. Bins within the function.

    :param list of int qs: The integrated charges from each individual waveform.
    """
    pass

def quick_qint(wform):
    """
    Finds the integral of a pulse, defined by a window around the global
    minimum in the waveform.

    :param np.array wform: Numpy array of the waveform to fit.
    """
    # Take the argmin as the peak
    peak_i = np.argmin(wform)

    # Halfsize of window in indices (NOT time)
    win_halfsize = 5

    # Define window around centre as peak, size determined by eye.
    peak_wform = wform[(peak_i-win_halfsize):(peak_i+win_halfsize)]

    # TODO: Deal with afterpulses
    # Get baseline from average of all points outside window
    non_peak = np.append(wform[:peak_i-win_halfsize], wform[:peak_i+win_halfsize])
    baseline = sum(non_peak)/len(non_peak)

    # Integrate Q from within window, considering baseline
    # Effectively flip, offset to 0, integrate
    peak_wform_mod = [baseline-x for x in peak_wform]
    qint = sum(peak_wform_mod)*digi_res

    return qint

def scan_qint(wform):
    # Slide window over whole waveform, integrating
    # Take the largest (actually smallest) window, define that as integral
    # Take basline outside window, minus off INTEGRATED baseline

    pass

def process_wforms(fname):
    """
    Opens CAEN digitizer XML output.
    Combines traces to to return average waveform.

    :param str fname: filename of input XML.
    """
    print("File: %s" % fname)
    split_fname = splitext(fname)

    if split_fname[1] == ".pkl":
        # If pickle file is passed just load wform list from that
        print("Loading from Pickle...")
        with open(fname, "rb") as f:
            wforms = pickle.load(f)
        print("... done! %i waveforms loaded." % len(wforms))
    else:
        print("Parsing XML...")
        tree = ET.parse(fname)
        root = tree.getroot()
        print("... done!")

        print("Loading waveforms...")
        wforms = []
        # Loops through every "event" in the XML, each with a "trace" (waveform)
        for i,child in enumerate(root.iter("event")):
            # Trace is spaced wform values. Split on spaces and convert to np array.
            # Use int as dtype to ensure numpy arithmetic.
            wform = np.array(child.find("trace").text.split()).astype(int)
            wforms.append(wform)
            if (i % 100) == 0:
                print("    %i loaded...\r" % i, end="")
        print("... done! %i waveforms loaded." % i)

        # Dump waveforms to pickle (quicker than parsing XML each time)
        pickle_fname = split_fname[0]+".pkl"
        with open(pickle_fname, "wb") as f:
            pickle.dump(wforms, f)
        print("Saved to file %s." % pickle_fname)

    # Average waveform
    wform_avg = sum(wforms)/len(wforms)
    
    # Pull integrated charges from file if it exists
    q_fname = split_fname[0] + "_q.pkl"
    if exists(q_fname):
        with open(q_fname, "rb") as f:
            qs = pickle.load(f)
        return qs, wform_avg

    # Otherwise, calculate
    qs = []
    print("Finding charge integrals...")
    for i,wform in enumerate(wforms):
        try:
            qs.append(quick_qint(wform))
        except IndexError:
            continue
        if (i % 100) == 0:
            print("    %i calculated...\r" % i, end="")
    print("... done! %i calculated." % i)
    with open(q_fname, "wb") as f:
        pickle.dump(qs, f)
    print("Saved to file %s." % q_fname)

    return qs, wform_avg

def main():
    # Use glob to expand wildcards
    fnames = []
    for arg in argv[1:]:
        fnames += glob(arg)

    for fname in fnames:
        # Keep American spelling for consistency...
        qs, wform_avg = process_wforms(fname)

        # Fit the integrated charge histo
        fit_qhist(qs)
        exit()

        # 1st figure is plot of peak centres
        plt.figure(1)
        plt.hist(qs, bins=qhist_bins, label=fname)

        # 2nd figure is the averaged waveform
        plt.figure(2)
        # Scale xs to match resolution
        xs = [digi_res*x for x in range(len(wform_avg))]
        plt.scatter(xs, wform_avg, marker="+")
        result = fit_wform(wform_avg)
        plt.plot(xs, result.best_fit, label=fname)
    plt.figure(1)
    plt.legend()
    plt.xlabel("t [ns]")
    plt.yscale("log")

    plt.figure(2)
    plt.legend()
    plt.xlabel("t [ns]")
    plt.ylabel("V [mV]")
    plt.show()

if __name__ == "__main__": main()