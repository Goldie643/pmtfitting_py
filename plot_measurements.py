import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from pathlib import Path

brand_dict = {"ham": "Hamamatsu", "nnvt" : "NNVT"}

fname = argv[1]

df = pd.read_csv(fname)

# Assumes standarde file naming format
def get_v_from_fname(row):
    base = Path(row["fname"]).stem

    # Get brand model and voltage from filename
    brand, model, voltage = base.split("_")
    brand = brand_dict[brand]
    # Get rid of "v" in string
    voltage = float(voltage[:-1])
    return brand, model, voltage

# Expand here means when apply returns multiple values, they get assigned to
# each given new column
df[["brand","model","v"]] = df.apply(get_v_from_fname, axis="columns",
    result_type="expand")

# Sort so V is in order
df = df.sort_values(["v","fname"], ascending=True)
df = df.set_index("v")

# Group by the model and plot each value
df_group = df.groupby("model")

plot_cols = {
    "gain" : "Gain", 
    "pv_r" : "Peak-Valley Ratio", 
    "sigma" : r"$\sigma$", 
    "pe_res" : "PE Resolution"
}

for key,value in plot_cols.items():
    fig, ax = plt.subplots()
    df_group[key].plot(legend=True)
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel(value)

plt.show()