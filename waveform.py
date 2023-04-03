import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

in_file = argv[1]

print("Parsing XML...", end="")
tree = ET.parse(in_file)
print(" done!")
print("Getting root...", end="")
root = tree.getroot()
print(" done!")

wform_tot = None
i = 0
for child in root.iter("event"):
    # print(child.tag, child.attrib)
    wform = np.array(child.find("trace").text.split()).astype(int)
    if wform_tot is None:
        wform_tot = wform
    else:
        wform_tot += wform
    i += 1
    if (i>10): break

wform_avg = wform_tot/i

plt.plot(range(len(wform_avg)), wform_avg)
# plt.plot(range(len(wform_tot)), wform_tot)
plt.show()