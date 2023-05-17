import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.signal
from glob import glob
from sys import argv
from os.path import splitext, exists
from lmfit.models import ConstantModel, GaussianModel

# Resolution of the CAEN digitiser
digi_res = 4 # ns
qhist_bins = 400 # Numbe of bins to use when fitting and histing qint

def fit_wform(wform):
    """
    Fits the waveform, assuming linear background and gaussian peak.

    :param np.array wform: Numpy array of the waveform to fit.
    """
    mod_bg = ConstantModel(prefix="bg_")
    mod_peak = GaussianModel(prefix="g1_")

    model = mod_bg + mod_peak

    # Guess the center as the global minimum, scaled by digitiser res
    g1_center = digi_res*np.argmin(wform)

    # Very basic guess of amplitude as average
    bg_amplitude = sum(wform)/len(wform)

    model.set_param_hint("g1_center", value=g1_center)
    model.set_param_hint("g1_amplitude", value=-5, min=-1)
    model.set_param_hint("g1_sigma", value=5)
    model.set_param_hint("bg_amplitude", value=bg_amplitude)

    params = model.make_params()

    # try:
    #     params = model.make_params(g1_amplitude=-5, g1_center=g1_center, 
    #         g1_sigma=2, bg_amplitude=3600)
    # except:
    #     print("!!Issue setting wform fit model params!!")
    #     params = model.make_params()

    # Scale x to fit to real time values
    xs = [digi_res*x for x in range(len(wform))]
    result = model.fit(wform, params, x=xs)

    # print(result.fit_report())

    return result

def find_peaks(qs, distance=200):
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

    peaks = scipy.signal.find_peaks(qs_hist[0], prominence=max(qs)/10)[0]
    print("%i peak(s) found with indices: " % len(peaks), end="")
    print(peaks)

    plt.vlines([peak*bin_width for peak in peaks], 0, max(qs), colors="r")
    plt.show()

    return

def fit_qhist(qs, npe=2, peak_spacing=400, peak_width=20):
    """
    Fits Gaussians to the integrated charge histogram, fitting the pedestal, 1pe
    and 2pe peaks. Bins within the function.

    :param list of int qs: The integrated charges from each individual waveform.
    :param int npe: The number of pe peaks to fit (not including pedestal).
    :param int or float peak_spacing: The guess at where the **1pe** peak will
        be. Subsequent pe peak guesses will be spaced equally apart.
    :param int or float peak_width: The guess at the sigma of the **1pe** peak.
        Subsequent peaks will be doubled.
    """
    
    # Bin the integrated charges 
    qs_hist, qs_binedges = np.histogram(qs, bins=qhist_bins)

    # Get centre of bins instead of edges
    bin_width = qs_binedges[1]-qs_binedges[0]

    # Includes all edges, get rid of last edge
    qs_bincentres = qs_binedges[:-1] + (bin_width/2)

    # Get rid of tiny bins by removing bins with less than qs_cut_thresh in
    qs_bincentres_cut = []
    qs_hist_cut = []
    qs_cut_thresh = max(qs_hist)/1e4
    for bincent, q in zip(qs_bincentres, qs_hist):
        if q > qs_cut_thresh:
            qs_bincentres_cut.append(bincent)
            qs_hist_cut.append(q)

    qs_hist = [x for x in qs_hist_cut]
    qs_bincentres = [x for x in qs_bincentres_cut]

    # Scale bin values to area normalise to 1
    qs_hist = qs_hist/(sum(qs_hist)*bin_width)

    # Linear flat background, gaussians for each peak
    # Don't currently use BG as it reduces effectiveness at fitting 2pe peak
    mod_bg = ConstantModel(prefix="bg_")

    # For some reason initial fit seems to be lower than it should be, scale it
    # up a bit to make it fit nicer initially
    scale = 20

    # Combine models
    model = mod_bg

    model.set_param_hint("bg_c", value=0, max=max(qs_hist)/1e4)

    # Find peaks, with fairly stringent prominence requirement
    peaks_i = scipy.signal.find_peaks(qs_hist, 
        prominence=max(qs_hist)/1000)[0]

    # Get actual peak positions instead of just indices
    peaks = [x*bin_width+qs_bincentres[0] for x in peaks_i]

    # TODO: instead, fit as many as npe, using spacing between previous peaks to
    # estimate position of next ones

    # Iteratively add npe pe peaks to fit
    for i in range(npe+1):
        model += GaussianModel(prefix=f"g{i}pe_")

        # First peak has width of peak_width, subsequent peaks will double in
        # width each time.
        # Will be overidden if peaks were found.
        width = peak_width*(2**i)

        if i < len(peaks):
            print("Peak Found")
            # If this peak was found, use them as starting guesses
            center = peaks[i]
            height = qs_hist[peaks_i[i]]

            # Use spacing if on at least 1pe peak
            if i > 0:
                print("Not ped")
                spacing = peaks[i] - peaks[i-1]
                width = spacing/4
        elif (len(peaks) > 1):
            print("No Peak Found")
            # TODO: ISSUE, THIS ASSUMES ITS ONLY ONE BEYOND PEAKS
            # Predict center using previous peak spacing
            prev_spacing = peaks[i-1] - peaks[i-2]
            center = peaks[i-1] + prev_spacing

            # Get height by getting the index, again from prev spacing
            prev_spacing_i = peaks_i[i-1] - peaks_i[i-2]
            height = qs_hist[peaks_i[i-1] + prev_spacing_i]

            # Assume width is double the previous peak spacing
            width = prev_spacing/2
        else:
            # Otherwise, just use the peak_spacing
            center = i*peak_spacing

            # Old method, predict based off of pedestal, not consistent
            # height = max(qs_hist)/(10**i)

            # Just take the height at the centre guess, get index from
            # converting peak_spacing into index spacing
            height = qs_hist[int(i*peak_spacing/bin_width)]

        print(width)

        # Scale to get to reasonable level
        height = height*scale

        model.set_param_hint(f"g{i}pe_center", value=center, min=0.9*center, 
            max=1.1*center)
        model.set_param_hint(f"g{i}pe_amplitude", value=height)

        model.set_param_hint(f"g{i}pe_sigma", value=width, min=0.75*width,
            max=1.5*width)

    # Make the params of the model
    params = model.make_params()

    # Scale x to fit to real time values
    qfit = model.fit(qs_hist, params, x=qs_bincentres)

    # Get each individual component of the model
    components = qfit.eval_components()

    qfit_fig, qfit_ax = plt.subplots()
    qfit_ax.bar(qs_bincentres, qs_hist, width=bin_width, label="Data", alpha=0.5)
    qfit_ax.plot(qs_bincentres, qfit.best_fit, label="Best Fit (Composite)")
    # Plot each component/submodel
    for name, sub_mod in components.items():
        try:
            # Get rid of underscore on prefix for submod name
            qfit_ax.plot(qs_bincentres, sub_mod, label=name[:-1])
        except ValueError:
            # For constant model, sub_mod isn't list
            # Don't use hlines, use plot to keep colours in order
            qfit_ax.plot([qs_bincentres[0],qs_bincentres[-1]], [sub_mod]*2, 
                label=name[:-1])

    qfit_ax.plot(qs_bincentres, qfit.init_fit, "--", c="grey", alpha=0.5, 
        label="Initial Fit")
    qfit_ax.vlines(peaks, 0, max(qs), colors="grey", linestyles="--", alpha=0.5)

    qfit_ax.legend()

    # Set lower limit to half a bin to avoid weird scaling
    # Remembering it's area normalised
    qfit_ax.set_ylim(bottom=0.1/len(qs))
    qfit_ax.set_yscale("log")

    return qfit, qs_hist, qs_bincentres, bin_width, qfit_ax, qfit_fig

def quick_qint(wform):
    """
    Finds the integral of a pulse, defined by a window around the global
    minimum in the waveform.

    :param np.array wform: Numpy array of the waveform to fit.
    """
    # The upper limit of the search region
    # Peaks are set to be early in the window
    search_region_lim = int(len(wform)/4)
    # Take the argmin as the peak
    peak_i = np.argmin(wform[:search_region_lim])

    # N bins pre/post the peak to calculate the integral from
    win_pre = 3
    win_post = 7

    # Define window around centre as peak, size determined by eye.
    peak_wform = wform[(peak_i-win_pre):(peak_i+win_post)]

    # TODO: Deal with afterpulses
    # Get baseline from average of all points outside window
    non_peak = np.append(wform[:peak_i-win_pre], wform[peak_i+win_post:])
    # Truncated mean, only use the middle 50% of values.
    non_peak.sort()
    non_peak_lim = int(len(non_peak)/4)
    non_peak = non_peak[non_peak_lim:-non_peak_lim]
    baseline = sum(non_peak)/len(non_peak)

    # plt.clf()
    # plt.plot(range(len(wform)), wform)
    # plt.plot(range(peak_i-win_pre,peak_i+win_post), peak_wform)
    # plt.plot([0,len(wform)], [baseline,baseline], c="grey", linestyle="--")
    # plt.show()

    # Integrate Q from within window, considering baseline
    # Effectively flip, offset to 0, integrate
    # Don't contribute negative charge to the integral.
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

def qint_calcs(qfit, qs_bincentres, qs_hist):
    """
    Calculates gain, peak-to-valley ratio and PE resolution from the
    integrated charge fit.

    :param lmfit.model.ModelResult qfit: The fit of the integrated charge histo.
    :param list or array of floats qs_bincentres: The centres of the qhist bins.
    :param list or array of floats qs_hist: The values of the qhist bins.
    """

    gped_center = qfit.best_values["g0pe_center"] # Or should this just be 0?
    g1pe_center = qfit.best_values["g1pe_center"]

    g1pe_amp = qfit.best_values["g1pe_amplitude"]
    g1pe_sig = qfit.best_values["g1pe_sigma"]

    g1pe_sig = qfit.best_values["g1pe_sigma"]

    two_peaks_fitted = "g2pe_center" in qfit.best_values

    # If there's a 2pe peak fit, check if it fit correctly (i.e. it is after the
    # 1pe peak)
    if two_peaks_fitted:
        g2pe_center = qfit.best_values["g2pe_center"]
        if g2pe_center < g1pe_center:
            print("ISSUE WITH FIT")
            print("2pe curve is centred BELOW 1pe.")
            return

    # Gain is just average integrated charge for 1pe vs none.
    gain = (g1pe_center - gped_center)/1.602e-19
    print(f"Gain = {gain:g}")

    # Peak-to-valley is ratio of 1pe peak to valley between 1pe and pedestal. 
    # Get valley from minimum of qs_hist between the two peaks
    # Subset qs_hist using qs_bincentres
    qs_hist_pedto1pe = [x for x in zip(qs_bincentres,qs_hist) 
        if x[0] > gped_center and x[0] < g1pe_center]
    # Get min from this subset
    if len(qs_hist_pedto1pe) == 0:
        print("Failed to calculate valley height!"
            " No data between ped and 1pe peak, maybe 1pe failed to fit?")
        h_v = None
    else:
        h_v = min([x[1] for x in qs_hist_pedto1pe])
    
    # Don't want to use the actual gaussian amplitude here as this is for peak
    # determination quality.
    # Find peak between the ped and 2pe centres (half the distance between them)
    qhist_1pe_peak_lo = (g1pe_center - gped_center)/2
    # Use second peak if it's fitted, if not just half beyond the 1pe peak
    if two_peaks_fitted:
        qhist_1pe_peak_hi = (g2pe_center - g1pe_center)/2
    else:
        qhist_1pe_peak_hi = 1.5*g1pe_center
    qs_hist_1pe_peak_scan = [x for x in zip(qs_bincentres,qs_hist) 
        if x[0] > qhist_1pe_peak_lo and x[0] < qhist_1pe_peak_hi]
    if len(qs_hist_1pe_peak_scan) == 0:
        print("Failed to calculate peak height!"
            " No data around 1pe peak, maybe 1pe failed to fit?")
        h_p = None
    else:
        h_p = max([x[1] for x in qs_hist_1pe_peak_scan])

    if h_p is not None and h_v is not None:
        pv_r = h_p/h_v
        print(f"Peak-to-valley ratio = {pv_r:g}")
    else:
        pv_r = None

    # PE resolution I *think* uses the actual fit gaussian
    pe_res = g1pe_sig/g1pe_amp
    print(f"PE Resolution = {pe_res:g}")

    return gain, pv_r, pe_res

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

        # Get calcs from qhist
        qint_calcs(qfit, qs_bincentres, qs_hist)

        # Plot integrated charges using the histogram info given by fit_qhist()
        qint_ax.bar(qs_bincentres, qs_hist, width=bin_width, alpha=0.5)
        qint_ax.plot(qs_bincentres, qfit.best_fit, label=fname)

        # Now plot average wform
        # Scale xs to match resolution
        xs = [digi_res*x for x in range(len(wform_avg))]
        
        # Get the fit of the waveform
        wform_fit = fit_wform(wform_avg)
        wform_fit_components = wform_fit.eval_components()

        # Offset by the fit BG ConstantModel
        offset_fit = [y-wform_fit_components["bg_"] for y in wform_fit.best_fit]
        offset_data = [y-wform_fit_components["bg_"] for y in wform_avg]

        # wform_ax.scatter(xs, offset_data, marker="+")
        wform_ax.plot(xs, offset_data, label=fname)
        # wform_ax.plot(xs, wform_fit.best_fit, label=fname)
        # wform_ax.plot(xs, offset_fit, label=fname)

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