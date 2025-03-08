# Elias Arnqvist, 2025

import uproot
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Read and plot some coincidences

file_name = "output/coincidences_co60_600s.root"
tree_name = "Data_C"

file = uproot.open(file_name)
tree = file[tree_name]

columns = ["Channel_a", "Channel_b", "Energy_a", "Energy_b", "Time_difference"]
data = tree.arrays(columns, library="np")

# Load calibration parameters
legal_channels = [1, 2, 3, 4]
cal_params = np.empty((max(legal_channels) + 1, 4), dtype=np.float64)
for i_ch, ch in enumerate(legal_channels):
    with open(f"calibration/ch{ch}.CALp", "r") as file:
        lines = file.readlines()
        row = np.array([ch,
                        lines[3].split(':')[1].strip(),
                        lines[2].split(':')[1].strip(),
                        lines[1].split(':')[1].strip()],
                       dtype=np.float64)
        cal_params[ch, :] = row

# Perform energy calibration
@jit("float64[:](int64[:], int64[:], float64[:,:])", nopython=True)
def energy_calibration(channels, energies, cal_params):
    energies_calibrated = np.zeros(len(energies))
    
    for i_event, (event_channel, event_energy) in enumerate(zip(channels, energies)):
        
        a, b, c = cal_params[event_channel, 1:]
        
        energies_calibrated[i_event] = a * event_energy**2 + b * event_energy + c
    
    return energies_calibrated

data["Energy_a_cal"] = energy_calibration(data["Channel_a"], data["Energy_a"], cal_params)
data["Energy_b_cal"] = energy_calibration(data["Channel_b"], data["Energy_b"], cal_params)

# %% Plot some results
plt.close('all')
inch_to_mm = 25.4
color = plt.cm.tab10

# Let us plot the energy-energy coincidence histogram between channels 3 and 4
sel1 = np.logical_and(data["Channel_a"] == 3, data["Channel_b"] == 4)
histo, ex, ey = np.histogram2d(data["Energy_a_cal"][sel1], 
                               data["Energy_b_cal"][sel1],
                               bins = (80, 80),
                               range = [[0, 1.6e3], [0, 1.6e3]])
fig, ax = plt.subplots(figsize=(100/inch_to_mm,90/inch_to_mm))
ax.pcolormesh(ex, ey, histo.T, norm=LogNorm(), cmap='viridis', rasterized=True)
ax.set_title("Coincidence events (channel 3 and 4)")
ax.set_xlabel("Energy channel 3 (keV)")
ax.set_ylabel("Energy channel 4 (keV)")
ax.set_aspect('equal')
# plt.tight_layout()
plt.tight_layout(pad = 0.2)
plt.subplots_adjust(hspace=0, wspace=0)

save_name = "co60_ch3ch4_energy3energy4"
plt.savefig(f'figures/{save_name}.jpg', dpi=300)
plt.savefig(f'figures/{save_name}.pdf')

plt.show()
