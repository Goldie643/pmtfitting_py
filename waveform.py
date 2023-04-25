import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
from sys import argv
from os.path import splitext, exists
from lmfit.models import LinearModel, ConstantModel, GaussianModel
from scipy.signal import find_peaks

# Resolution of the CAEN digitiser
digi_res = 4 # ns
qhist_bins = 200 # Numbe of bins to use when fitting and histing qint
peak_guess = [0, 250, 500, 750] # Guesses at where the ped, 1pe and 2pe peaks will be

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

    try:
        params = model.make_params(g1_amplitude=-20, g1_center=g1_center, 
            g1_sigma=2, lin_amplitude=120)
    except:
        print("!!Issue setting wform fit model params!!")
        params = model.make_params()

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
    
    # Bin the integrated charges 
    qs_hist, qs_binedges = np.histogram(qs, bins=qhist_bins)
    # Get centre of bins instead of edges
    bin_width = qs_binedges[1]-qs_binedges[0]

    # Includes all edges, get rid of last edge
    qs_bincentres = qs_binedges[:-1] + (bin_width/2)

    # Scale bin values to area normalise to 1
    qs_hist = qs_hist/(sum(qs_hist)*bin_width)

    # Linear flat background, gaussians for each peak
    # Don't currently use BG as it reduces effectiveness at fitting 2pe peak
    mod_bg = ConstantModel(prefix="bg_")
    mod_ped = GaussianModel(prefix="gped_")
    mod_1pe = GaussianModel(prefix="g1pe_")
    mod_2pe = GaussianModel(prefix="g2pe_")
    mod_3pe = GaussianModel(prefix="g3pe_")

    # Combine all Gaussians for fit
    model = mod_bg + mod_ped + mod_1pe + mod_2pe + mod_3pe

    # Pedestal should be highest peak
    gped_amp_guess = 2*max(qs_hist)
    # For usual LED settings, subsequent peaks should be smaller
    # Halving is just a guess
    g1pe_amp_guess = gped_amp_guess/2
    g2pe_amp_guess = g1pe_amp_guess/2
    g3pe_amp_guess = g2pe_amp_guess/2

    model.set_param_hint("gped_center", value=peak_guess[0], min=-5, max=5)
    model.set_param_hint("g1pe_center", value=peak_guess[1])
    model.set_param_hint("g2pe_center", value=peak_guess[2])
    model.set_param_hint("g3pe_center", value=peak_guess[3])
    model.set_param_hint("gped_sigma", value=5)
    model.set_param_hint("g1pe_sigma", value=50)
    model.set_param_hint("g2pe_sigma", value=100)
    model.set_param_hint("g3pe_sigma", value=200)
    model.set_param_hint("bg_c", value=1e-5)
    params = model.make_params()
    # params = model.make_params(
    #     gped_center=peak_guess[0],
    #     g1pe_center=peak_guess[1],
    #     g2pe_center=peak_guess[2],
    #     gped_amplitude=gped_amp_guess,
    #     g1pe_amplitude=g1pe_amp_guess,
    #     g2pe_amplitude=g2pe_amp_guess,
    #     # These are again, general guesses
    #     gped_sigma=5,
    #     g1pe_sigma=50,
    #     g2pe_sigma=100
    # )

    # Scale x to fit to real time values
    qfit = model.fit(qs_hist, params, x=qs_bincentres)

    # Get each individual component of the model
    components = qfit.eval_components()

    qfit_fig, qfit_ax = plt.subplots()
    qfit_ax.bar(qs_bincentres, qs_hist, width=bin_width, label="Data", alpha=0.5)
    # qfit_ax.hist(qs, bins=qhist_bins, label="Data", alpha=0.5, density=True)
    qfit_ax.plot(qs_bincentres, qfit.init_fit, "--", c="grey", alpha=0.5)
    qfit_ax.plot(qs_bincentres, qfit.best_fit, label="Best Fit (Composite)")
    # Plot each component/submodel
    for name, sub_mod in components.items():
        try:
            # Get rid of underscore on prefix for submod name
            qfit_ax.plot(qs_bincentres, sub_mod, label=name[:-1])
        except ValueError:
            # For constant model, sub_mod isn't list
            qfit_ax.hlines(y=sub_mod, xmin=qs_bincentres[0], xmax=qs_bincentres[-1], 
                label=name[:-1])
    qfit_ax.legend()

    # Set lower limit to half a bin to avoid weird scaling
    # Remembering it's area normalised
    qfit_ax.set_ylim(bottom=0.5/len(qs))
    qfit_ax.set_yscale("log")

    return qfit, qs_hist, qs_bincentres, bin_width, qfit_ax, qfit_fig

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

    # Show a few example waveforms
    # for wform in wforms[:3]:
    #     plt.plot(range(len(wform)), wform)
    #     plt.show()
    
    # Pull integrated charges from file if it exists
    q_fname = split_fname[0] + ".qpkl"
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

    # Set up plotting figs/axes
    qint_fig, qint_ax = plt.subplots()
    wform_fig, wform_ax = plt.subplots()

    for fname in fnames:
        # Keep American spelling for consistency...
        qs, wform_avg = process_wforms(fname)

        # Fit the integrated charge histo
        qfit, qs_hist, qs_bincentres, bin_width, qfit_ax, qfit_fig = fit_qhist(qs)
        qfit_ax.set_title(fname)

        # Plot integrated charges using the histogram info given by fit_qhist()
        qint_ax.bar(qs_bincentres, qs_hist, width=bin_width, alpha=0.5)
        qint_ax.plot(qs_bincentres, qfit.best_fit, label=fname)

        # Now plot average wform
        # Scale xs to match resolution
        xs = [digi_res*x for x in range(len(wform_avg))]
        wform_ax.scatter(xs, wform_avg, marker="+")
        wform_fit = fit_wform(wform_avg)
        wform_ax.plot(xs, wform_fit.best_fit, label=fname)

    qint_ax.legend()
    qint_ax.set_yscale("log")
    # Set lower limit to half a bin to avoid weird scaling
    # Remembering it's area normalised
    qint_ax.set_ylim(bottom=0.5/len(qs))
    qint_ax.set_xlabel("Integrated Charge")

    wform_ax.legend()
    wform_ax.set_xlabel("t [ns]")
    wform_ax.set_ylabel("V [mV]")

    plt.show()

if __name__ == "__main__": main()