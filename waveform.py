import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
from sys import argv
from os.path import splitext
from lmfit.models import LinearModel, GaussianModel

# Resolution of the CAEN digitiser
digi_res = 4 # ns

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
    
    # Fitting every waveform, far too slow
    #
    # fits = []
    # failed_fits = 0
    # # Fit wforms, add gaussian centre to histo
    # print("Fitting waveforms...")
    # for i,wform in enumerate(wforms):
    #     fit = fit_wform(wform)
    #     if fit.success:
    #         fits.append(fit)
    #     else:
    #         failed_fits += 1

    #     if (i % 100) == 0:
    #         print("    %i fitted, %i failed...\r" % (i,failed_fits), end="")
    # print("... done! %i waveforms fitted, %i failed." % (i,failed_fits))

    # Take the minimum value in the waveform as the centre of the pulse
    centers = [digi_res*np.argmin(wform) for wform in wforms]

    # Dump waveforms to pickle (quicker than parsing XML each time)
    pickle_fname = split_fname[0]+".pkl"
    with open(pickle_fname, "wb") as f:
        pickle.dump(wforms, f)

    wform_avg = sum(wforms)/len(wforms)

    return centers, wform_avg

def main():
    # Use glob to expand wildcards
    fnames = []
    for arg in argv[1:]:
        fnames += glob(arg)

    for fname in fnames:
        # Keep American spelling for consistency...
        centers, wform_avg = process_wforms(fname)

        # Getting centre of fits is too slow
        # can optimise but for now will use argmin
        #
        # Get the centre of every fit, if it's within the range
        # centers = []
        # for fit in fits:
        #     center = fit.params["g1_center"].value 
        #     if center < 1200:
        #         centers.append(center)

        # 1st figure is plot of peak centres
        plt.figure(1)
        plt.hist(centers, bins=100)

        # 2nd figure is the averaged waveform
        plt.figure(2)
        # Scale xs to match resolution
        xs = [digi_res*x for x in range(len(wform_avg))]
        plt.scatter(xs, wform_avg, marker="+")
        result = fit_wform(wform_avg)
        plt.plot(xs, result.best_fit, label=fname)
    plt.figure(1)
    plt.xlabel("t [ns]")
    plt.xlim([0,1200])
    plt.yscale("log")

    plt.figure(2)
    plt.legend()
    plt.xlabel("t [ns]")
    plt.ylabel("V [mV]")
    plt.show()

if __name__ == "__main__": main()