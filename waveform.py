import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sys import argv

# Use glob to expand wildcards
fnames = []
for arg in argv[1:]:
    fnames += glob(arg)

def combine_wforms(fname):
    """
    Opens CAEN digitizer XML output.
    Combines traces to to return total and average waveform

    :param str fname: filename of input XML.
    """
    print("File: %s")
    print("Parsing XML..." % fname)
    tree = ET.parse(fname)
    root = tree.getroot()
    print("... done!")

    print("Processing waveforms...")

    wforms = []
    # Loops through every "event" in the XML, each with a "trace" (waveform)
    for i,child in enumerate(root.iter("event")):
        # Trace is spaced wform values, split on spaces and convert to np array.
        # Use int as dtype to ensure numpy arithmetic.
        wforms.append(np.array(child.find("trace").text.split()).astype(int))
        if (i % 100) == 0:
            print("    %i processed...\r" % i, end="")

    wform_tot = sum(wforms)
    wform_avg = wform_tot/i
    print("... done! %i waveforms processed" % i)

    return (wform_avg, wform_tot)

# TODO: Separate total and average plots

for fname in fnames:
    wform_avg, wvform_tot = combine_wforms(fname)
    plt.plot(range(len(wform_avg)), wform_avg, label=fname)
# plt.plot(range(len(wform_tot)), wform_tot)
plt.legend()
plt.show()