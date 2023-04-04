import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pickle
from glob import glob
from sys import argv
from os.path import splitext

# Use glob to expand wildcards
fnames = []
for arg in argv[1:]:
    fnames += glob(arg)

print(fnames)

def combine_wforms(fname):
    """
    Opens CAEN digitizer XML output.
    Combines traces to to return average waveform.

    :param str fname: filename of input XML.
    """
    print("File: %s" % fname)
    split_fname = splitext(fname)

    # If pickle file is passed just load wform list from that
    if split_fname[1] == ".pkl":
        print("Loading from Pickle...")
        with open(fname, "rb") as f:
            wforms = pickle.load(f)
        wform_avg = sum(wforms)/len(wforms)
        print("... done!")
        return wform_avg

    print("Parsing XML...")
    tree = ET.parse(fname)
    root = tree.getroot()
    print("... done!")

    print("Processing waveforms...")
    wforms = []
    # Loops through every "event" in the XML, each with a "trace" (waveform)
    for i,child in enumerate(root.iter("event")):
        # Trace is spaced wform values. Split on spaces and convert to np array.
        # Use int as dtype to ensure numpy arithmetic.
        wforms.append(np.array(child.find("trace").text.split()).astype(int))
        if i>10: break
        if (i % 100) == 0:
            print("    %i processed...\r" % i, end="")

    # Dump waveforms to pickle (quicker than parsing XML each time)
    pickle_fname = split_fname[0]+".pkl"
    with open(pickle_fname, "wb") as f:
        pickle.dump(wforms, f)

    wform_avg = sum(wforms)/len(wforms)
    print("... done! %i waveforms processed." % i)

    return wform_avg

for fname in fnames:
    wform_avg = combine_wforms(fname)
    plt.plot(range(len(wform_avg)), wform_avg, label=fname)
plt.legend()
plt.show()