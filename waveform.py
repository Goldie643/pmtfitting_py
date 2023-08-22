import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.signal
import pandas as pd
from glob import glob
from sys import argv
from os.path import splitext, exists
from lmfit.models import ConstantModel, GaussianModel
from lmfit import Model
from datetime import datetime as dt

# Resolution of the CAEN digitiser
digi_res = 4 # ns
qhist_bins = 500 # Number of bins to use when fitting and histing qint
peak_prominence_fac = 200 # Prominance will be max value of hist/this

# What to scale the gain calc by (to get diff units)
# Digitiser resolution in ns, switch to s
# Switch mV to V
# Divide by e to get to gain
gain_scale = digi_res*1e-9*1e-3/1.602e-19

# Resistance of cable
gain_scale /= 50 # Ohm

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

def fit_qhist(qs, npe=2):
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

    # Get rid of qs where calculation is perfectly 0 for some reason
    qs_filter = [q for q in qs if q != 0]
    
    # Bin the integrated charges 
    qs_hist, qs_binedges = np.histogram(qs_filter, bins=qhist_bins)

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

    # Exponential aiming to model the under-amplified signals where electrons
    # skip a dynode. lmfit's exp model isn't the right format, make our own.
    def exp_bg(x, alpha):
        return alpha*np.exp(-alpha*x)
    mod_exp = Model(exp_bg, prefix="exp_")

    # Combine models
    model = mod_bg + mod_exp

    model.set_param_hint("bg_c", value=0, max=max(qs_hist)/1e4)
    model.set_param_hint("exp_alpha", value=max(qs_hist)/20)

    # Find peaks, with fairly stringent prominence requirement, and distance
    # being greater than 0.1 the total span of the hist.
    peaks_i_tmp = scipy.signal.find_peaks(qs_hist, 
        prominence=max(qs_hist)/peak_prominence_fac,
        distance = 0.1*len(qs_hist))[0]

    peaks_i = peaks_i_tmp

    # Get actual peak positions instead of just indices
    peaks = [x*bin_width+qs_bincentres[0] for x in peaks_i]

    # Use found peaks to estimate pedestal width and spacing
    # This may be overwritten
    if len(peaks) > 1:
        peak_width = 0.05*(peaks[1] - peaks[0])
        peak_spacing = peaks[1] - peaks[0]

        # The limits on centre, tight if peaks are found
        # As a factor on the value itself
        min_centre = 0.9
        max_centre = 1.1

        min_width = 0.5
    else:
        peak_width = 0.02*max(qs_bincentres)
        peak_spacing = 0.3*max(qs_bincentres)

        # Loosen limits on centre if only one peak found
        min_centre = 0.5
        max_centre = 2

        min_width = 0

    max_widths = [4, 1.2]
    max_widths.extend([2]*(npe-1))

    # Iteratively add npe pe peaks to fit
    for i in range(npe+1):
        if model is None:
            model = GaussianModel(prefix=f"g{i}pe_")
        else:
            model += GaussianModel(prefix=f"g{i}pe_")

        # First peak has width of peak_width, subsequent peaks will double in
        # width each time.
        # Will be overidden if peaks were found.
        width = peak_width*(2**i)

        # If beyond the peaks, let the width go as wide as it likes
        # if i > (len(peaks) - 1):
        #     min_width = 0
        #     min_width = np.inf

        #     min_centre = 0
        #     min_centre = 10


        if i < len(peaks):
            # If this peak was found, use them as starting guesses
            center = peaks[i]
            # height = qs_hist[peaks_i[i]]

            # Use spacing if on at least 1pe peak
            if i > 0:
                spacing = peaks[i] - peaks[i-1]
                width = spacing/3
        elif (len(peaks) > 1) and (i == len(peaks)):
            # If the peak to fit is one beyond the peaks found,
            # predict center using previous peak spacing
            prev_spacing = peaks[i-1] - peaks[i-2]
            center = peaks[i-1] + prev_spacing

            # Get height by getting the index, again from prev spacing
            prev_spacing_i = peaks_i[i-1] - peaks_i[i-2]
            # height = qs_hist[peaks_i[i-1] + prev_spacing_i]

            # Assume width is double the previous peak spacing
            width = prev_spacing/2
        else:
            # Otherwise, just use the peak_spacing
            center = i*peak_spacing


        # Just take the height at the centre guess, get index from
        # converting peak_spacing into index spacing
        # height = qs_hist[int(center/bin_width)]
        center_i_calc = int(center/bin_width - qs_bincentres[0])
        height = qs_hist[center_i_calc]

        # Calc amplitude from gaussian calc. 
        # Doesn't perfectly line up but oh well.
        amp = height*width*np.sqrt(2*np.pi)

        # model.set_param_hint(f"g{i}pe_center", value=center)
        model.set_param_hint(f"g{i}pe_center", value=center, 
            min=min_centre*center, max=max_centre*center)

        # Hinting at height, not amplitude, doesn't work accurately for some reason
        model.set_param_hint(f"g{i}pe_amplitude", value=amp)

        # model.set_param_hint(f"g{i}pe_sigma", value=width)
        # model.set_param_hint(f"g{i}pe_sigma", value=width, 
        #     min=min_width*width, max=max_width*width)
        model.set_param_hint(f"g{i}pe_sigma", value=width, 
            min=min_width*width, max=max_widths[i]*width)

    # Make the params of the model
    params = model.make_params()

    # Scale x to fit to real time values
    qfit = model.fit(qs_hist, params, x=qs_bincentres)

    # Get each individual component of the model
    components = qfit.eval_components()

    qfit_fig, qfit_ax = plt.subplots()
    qfit_fig.set_size_inches(14,8)

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

    # qfit_ax.plot(qs_bincentres, qfit.init_fit, "--", c="grey", alpha=0.5, 
    #     label="Initial Fit")
    qfit_ax.vlines(peaks, 0, max(qs_hist), colors="grey", linestyles="--", 
        alpha=0.5)

    qfit_ax.legend()

    # Set lower limit to half a bin to avoid weird scaling
    # Remembering it's area normalised
    qfit_ax.set_ylim(bottom=0.1/len(qs))
    qfit_ax.set_yscale("log")

    # Have to return ax and fig to set title to fname.
    # Or could take plot title here instead.
    return qfit, qs_hist, qs_bincentres, peaks_i, qfit_ax, qfit_fig

def quick_qint(wform, vbin_width=1):
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
    # plt.plot(range(len(wform)), wform, label="Outside Pulse")
    # plt.plot(range(peak_i-win_pre,peak_i+win_post), peak_wform, 
    #     label="Peak (Integral region)")
    # plt.plot([0,len(wform)], [baseline,baseline], c="grey", linestyle="--", 
    #     label="Calculated Baseline")
    # plt.legend()
    # plt.show()

    # Integrate Q from within window, considering baseline
    # Effectively flip, offset to 0, integrate
    # Don't contribute negative charge to the integral.
    peak_wform_mod = [baseline-x for x in peak_wform]
    qint = sum(peak_wform_mod)*vbin_width

    return qint

def load_wforms(fname):
    """
    Opens CAEN digitizer XML output and returns waveforms and digitiser 
    setttings.

    :param str fname: filename of input XML.
    """
    print("File: %s" % fname)
    split_fname = splitext(fname)

    if split_fname[1] == ".pkl":
        # If pickle file is passed just load wform list from that
        # Voltage range and resolution should also be stored in pkl
        print("Loading from Pickle...")
        with open(fname, "rb") as f:
            wforms, vrange, vbin_width = pickle.load(f)
        print("... done! %i waveforms loaded." % len(wforms))

        return wforms, vrange, vbin_width

    print("Parsing XML...")
    tree = ET.parse(fname)
    root = tree.getroot()
    print("... done!")

    # Get voltage range and resolution
    digi = root.find("digitizer")
    vrange_xml = digi.find("voltagerange").attrib
    # Convert strings to floats for hi and low.
    vlow = float(vrange_xml["low"])
    vhi = float(vrange_xml["hi"])
    vrange = (vlow, vhi)

    res = digi.find("resolution").attrib["bits"]
    # Convert bins to charge
    # Get voltage range from digitiser, divide by number of bins (2^number
    # of bits). 
    vbin_width = (vrange[1]-vrange[0])/(2**int(res))
    # Convert to mV
    vbin_width *= 1e3

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
        pickle.dump((wforms,vrange,vbin_width), f)
    print("Saved to file %s." % pickle_fname)

    return wforms, vrange, vbin_width

def process_wforms_q(wforms, split_fname, vbin_width):
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
            qs.append(quick_qint(wform,vbin_width))
        except IndexError:
            continue
        if (i % 100) == 0:
            print("    %i calculated...\r" % i, end="")
    print("... done! %i calculated." % i)
    with open(q_fname, "wb") as f:
        pickle.dump(qs, f)
    print("Saved to file %s." % q_fname)

    return qs, wform_avg

def process_wforms_dr(wforms, vbin_width):
    n_thresh = 20
    thresholds = np.linspace(-0.5,-2.5, n_thresh)
    passes = [0]*len(thresholds)

    import time
    start = time.time()
    for i,wform in enumerate(wforms):
        print(f"Waveforms checked: {i}\r",end="")

        wform_trunc = wform.copy()
        # Truncated mean, only use the middle 50% of values to find baseline
        wform_trunc.sort()
        wform_trunc_lim = int(len(wform_trunc)/4)
        wform_trunc = wform_trunc[wform_trunc_lim:-wform_trunc_lim]
        baseline = sum(wform_trunc)/len(wform_trunc)

        # Offset to 0 and scale to be voltage
        # Using np array is much quicker here
        wform_offset = np.array(wform, dtype=np.float64)
        wform_offset -= baseline
        wform_offset *= vbin_width
        wform_min = min(wform_offset)

        for i,threshold in enumerate(thresholds):
            if wform_min < threshold:
                passes[i] += 1
    print(f"Time taken: {time.time() - start}")

    passes_frac = [x/len(wforms) for x in passes]

    # TODO: convert this into an actual dark rate.
    # Need to count peaks that pass threshold in window, get window length.
    # Currently we are effectively assuming a maximum of one peak per window.
    plt.plot(thresholds, passes_frac)
    plt.scatter(thresholds, passes_frac, marker="x")
    plt.show()

    # pass_percent = n_passes*100.0/len(wforms)

    # print(f"{n_passes} of {len(wforms)} wforms passed threshold ({pass_percent}%).")

    return

def qint_calcs_fit(qfit, qs_bincentres, qs_hist):
    """
    Calculates gain and peak-to-valley ratio from the integrated charge fit.

    :param lmfit.model.ModelResult qfit: The fit of the integrated charge histo.
    :param list or array of floats qs_bincentres: The centres of the qhist bins.
    :param list or array of floats qs_hist: The values of the qhist bins.
    """

    gped_center = qfit.best_values["g0pe_center"] # Or should this just be 0?
    g1pe_center = qfit.best_values["g1pe_center"]

    two_peaks_fitted = "g2pe_center" in qfit.best_values

    # If there's a 2pe peak fit, check if it fit correctly (i.e. it is after the
    # 1pe peak)
    if two_peaks_fitted:
        g2pe_center = qfit.best_values["g2pe_center"]
        if g2pe_center < g1pe_center:
            print("ISSUE WITH FIT")
            print("2pe curve is centred BELOW 1pe.")
            return -1, -1

    # Gain is just average integrated charge for 1pe vs none.
    gain = (g1pe_center - gped_center)
    gain *= gain_scale
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

    return gain, pv_r

def qint_calcs_peaks(peaks_i, qs_bincentres, qs_hist):
    """
    Calculates gain and peak-to-valley ratio from the peaks found (*not* the
    gaussian fits, just the peaks).

    :param list or array of ints peaks_i: The indices of the peaks in qhist,
        starting with the pedestal (0pe).
    :param list or array of floats qs_bincentres: The centres of the qhist bins.
    :param list or array of floats qs_hist: The values of the qhist bins.
    """
    if len(peaks_i) < 2:
        print("Less than 2 peaks, can't make gain/PV calcs using peaks.")
        return None, None

    bin_width = qs_bincentres[1] - qs_bincentres[0]

    # Gain is just average integrated charge for 1pe vs none.
    gain = (peaks_i[1] - peaks_i[0])*bin_width
    gain *= gain_scale
    print(f"Gain = {gain:g}")

    # Valley between pedestal and 1pe peak
    valley = min(qs_hist[peaks_i[0]:peaks_i[1]])

    # Ratio of 1pe peak to valley
    pv_r = qs_hist[peaks_i[1]]/valley
    print(f"Peak-to-valley ratio = {pv_r:g}")

    return gain, pv_r

def qint_calcs(qfit, peaks_i, qs_bincentres, qs_hist):
    """
    Calculates gain and peak-to-valley ratio from the peaks and fits.

    :param lmfit.model.ModelResult qfit: The fit of the integrated charge histo.
    :param list or array of ints peaks_i: The indices of the peaks in qhist,
        starting with the pedestal (0pe).
    :param list or array of floats qs_bincentres: The centres of the qhist bins.
    :param list or array of floats qs_hist: The values of the qhist bins.
    """
    if len(peaks_i) >= 2:
        gain, pv_r = qint_calcs_peaks(peaks_i, qs_bincentres, qs_hist)
    else:
        gain, pv_r = qint_calcs_fit(qfit, qs_bincentres, qs_hist)
        
    g1pe_amp = qfit.best_values["g1pe_amplitude"]
    g1pe_sig = qfit.best_values["g1pe_sigma"]

    print(f"1PE Sigma = {g1pe_sig:g}")
        
    # PE resolution uses the actual fit gaussian
    pe_res = g1pe_sig/g1pe_amp
    print(f"PE Resolution = {pe_res:g}")

    return gain, pv_r, g1pe_sig, pe_res

def process_files_q(fnames):
    # Set up plotting figs/axes
    qint_fig, qint_ax = plt.subplots()
    wform_fig, wform_ax = plt.subplots()

    qint_fig.set_size_inches(14,8)
    wform_fig.set_size_inches(14,8)

    chisqrs = []
    gains = []
    pv_rs = []
    sigs = []
    pe_ress = []

    for fname in fnames:
        split_fname = splitext(fname)

        # Keep American spelling for consistency...
        wforms, vrange, vbin_width = load_wforms(fname)
        qs, wform_avg = process_wforms_q(wforms, split_fname, vbin_width)

        # Fit the integrated charge histo
        qfit, qs_hist, qs_bincentres, peaks_i, qfit_ax, qfit_fig = fit_qhist(qs)
        print(f"Chisqr = {qfit.chisqr:g}")
        bin_width = qs_bincentres[1] - qs_bincentres[0]
        qfit_ax.set_title(fname)

        if "--save_plots" in argv:
            qfit_fig.savefig(f"{split_fname[0]}_qint.pdf")

        # Get calcs from qhist
        # qint_calcs_fit(qfit, qs_bincentres, qs_hist)

        # Calculate based on the peak finder
        gain, pv_r, g1pe_sig, pe_res = qint_calcs(qfit, peaks_i, 
            qs_bincentres, qs_hist)
        if gain == -1 and pv_r == -1:
            print("Issues with fit mean file will be skipped.")
        chisqrs.append(qfit.chisqr)
        gains.append(gain)
        pv_rs.append(pv_r)
        sigs.append(g1pe_sig)
        pe_ress.append(pe_res)

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

    now_str = dt.now().strftime("%Y%m%d%H%M%S")

    # Either saves or shows the plots
    if "--save_plots" in argv:
        qint_fig.savefig(f"{now_str}_qint.pdf")
        wform_fig.savefig(f"{now_str}_wform.pdf")
    elif "--show_plots" in argv:
        plt.show()

    # Don't save data if flag not given
    # TODO: Use actual argparse
    if "--save" not in argv:
        return

    # Dump to csv via pandas
    # Could do it manually but plotter already uses pandas and hey who doesn't
    # love pandas, everyone should have it available.
    calcs = {
        "fname": fnames,
        "chisqr": chisqrs,
        "gain": gains,
        "pv_r": pv_rs,
        "sigma": sigs,
        "pe_res": pe_ress
    }
    calcs_df = pd.DataFrame.from_dict(calcs)
    csv_name = f"{now_str}_pmt_measurements.csv"
    calcs_df.to_csv(csv_name, index=False)
    return

def process_files_dr(fnames):
    for fname in fnames:
        wforms, vrange, vbin_width = load_wforms(fname)
        process_wforms_dr(wforms, vbin_width)

    return

def main():
    if "-h" in argv or "--help" in argv:
        print("--save       : Saves the fit information to csv.")
        print("--save_plots : Saves the qfit and wform plots to the input dir.")
        print("--show_plots : Shows the plots instead of saving them to file.")
        print("--q          : Integrate peaks to measure gain, PEres/sigma, PV ratio.")
        print("--dr         : Dark rate calculation, count peaks above thresholds.")
        print("WARNING: --show_plots and --save_figs cannot be used together."
            " Same for --q and --dr")
        return

    # Use glob to expand wildcards
    fnames = []
    for arg in argv[1:]:
        fnames += glob(arg)

    if len(fnames) == 0:
        print("Please give .xml or .pkl file to process.")
        return

    if "--q" in argv:
        process_files_q(fnames)
    elif "--dr" in argv:
        process_files_dr(fnames)
    else:
        print("Please pass --q (q integral) or --dr (dark rate) as an argument.")
        return

    return

if __name__ == "__main__": main()