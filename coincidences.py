import uproot
import numpy as np
from numba import jit, uint16, uint64


import time


# =============================================================================
# Settings
# =============================================================================

data_file = "data/big.root"
tree_name = "Data_R"




# =============================================================================
# Function
# =============================================================================

# Maximum time difference in ps between two events to be called coincident
time_max = np.uint64(250 * 1e3)

# @jit("uint16[:], uint64[:], uint16[:]", nopython=True)
# def process_data(channels, timestamps, energies):
#     # coincidence_events_chunk = np.empty((0, 5))
#     coincidence_events_chunk = []
    
#     for i_event, (event_channel, event_timestamp, event_energy) in enumerate(zip(channels, timestamps, energies)):
        
#         time_edge = event_timestamp + time_max
        
#         for i_other in range(i_event + 1, len(timestamps)):
#             other_timestamp = timestamps[i_other]
#             other_channel = channels[i_other]

#             if other_timestamp < time_edge and other_channel != event_channel:
#                 # Coincidence has occurred
                
#                 other_energy = energies[i_other]
                
#                 time_difference = np.int64(other_timestamp - event_timestamp)
                
#                 new_row = [event_channel, other_channel, event_energy, other_energy, time_difference]
#                 coincidence_events_chunk.append(new_row)
#                 new_row = [other_channel, event_channel, other_energy, event_energy, -time_difference]
#                 coincidence_events_chunk.append(new_row)
                
#             else:
#                 # Avoid looking at unneccessary data
#                 break
#     return coincidence_events_chunk


# new_event_chunk = process_data(data_np["Channel"], data_np["Timestamp"], data_np["Energy_cal"])
# new_event_chunk2 = np.array(new_event_chunk)


@jit("uint16[:], uint64[:], uint16[:]", nopython=True)
def process_data(
    channels: np.ndarray,    # uint16
    timestamps: np.ndarray,  # uint64
    energies: np.ndarray    # uint16
) -> np.ndarray:
    # Pre-allocate with estimated size
    n_events = len(timestamps)
    max_coincidences = n_events * 4
    
    # Use structured array for better memory layout and access
    result = np.zeros((max_coincidences, 5), dtype=np.uint64)
    counter = 0
    
    for i_event in range(n_events):
        event_channel = channels[i_event]
        event_timestamp = timestamps[i_event]
        event_energy = energies[i_event]
        
        time_edge = event_timestamp + time_max
        
        # Vectorized search for potential coincidences
        potential_indices = np.where(
            (timestamps[i_event+1:] < time_edge) & 
            (channels[i_event+1:] != event_channel)
        )[0] + i_event + 1
        
        for i_other in potential_indices:
            other_timestamp = timestamps[i_other]
            other_channel = channels[i_other]
            other_energy = energies[i_other]
            time_difference = other_timestamp - event_timestamp
            
            # Direct array assignment
            result[counter] = (event_channel, other_channel, 
                             event_energy, other_energy, 
                             time_difference)
            counter += 1
            
            # result[counter] = (other_channel, event_channel, 
            #                  other_energy, event_energy, 
            #                  np.uint64(-time_difference))
            # counter += 1
            
            if counter >= max_coincidences - 2:
                new_result = np.zeros((max_coincidences * 2, 5), dtype=np.uint64)
                new_result[:max_coincidences] = result
                result = new_result
                max_coincidences *= 2
    
    return result[:counter]





# =============================================================================
# Import data
# =============================================================================

buffer_size = "100 MB"

start_time = time.time()


illegal_flags = np.array([0x80, 0x400, 0x480], dtype=np.uint32)

legal_channels = np.array([1, 2, 3, 4], dtype=np.uint16)

legal_branches = ["Channel", "Energy", "Timestamp"]

events = uproot.open(data_file + ":" + tree_name)

for event_chunk in events.iterate(step_size=buffer_size, library="np"):    
    # Count the number of flags
    # flags, counts = np.unique(event_chunk["Flags"], return_counts=True)
    # flag_counts = {int(value): {'count': count, 'hex': hex(value)} for value, count in zip(flags, counts)}
    



    # Apply selections
    # selection = np.logical_and(np.isin(event_chunk["Flags"], illegal_flags, invert=True),
    #                             np.isin(event_chunk['Channel'], legal_channels))
    selection = np.isin(event_chunk["Flags"], illegal_flags, invert=True)
    # selection2 = np.isin(event_chunk['Channel'], legal_channels)
    # event_chunk = {key : value[selection] for key, value in event_chunk.items() if key in legal_branches}
    
    
    process_data(event_chunk["Channel"], event_chunk["Timestamp"], event_chunk["Energy"])
    
    pass
    
    


# data_np = tree.arrays(["Channel", "Timestamp", "Energy", "Flags"], library="np")




end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

#%%

























#%%

# Define dtype for structured array
cal_dtype = [('ch', np.uint16), ('a', np.float64), ('b', np.float64), ('c', np.float64)]
cal_params = np.zeros(len(legal_channels), dtype=cal_dtype)

# Read calibration files and fill structured array
for i, ch in enumerate(legal_channels):
    with open(f"cal_python\\ch{ch}.CALp", "r") as file:
        lines = file.readlines()
        # Remove trailing comma from tuple
        row = (np.uint16(ch),
               np.float32(lines[3].split(':')[1].strip()),
               np.float32(lines[2].split(':')[1].strip()),
               np.float32(lines[1].split(':')[1].strip()))
        cal_params[i] = row

# print(cal_params)

#%%

# Energy calibration
@jit(nopython=True)
def energy_calibration(channels, energies, cal_params):
    energies_calibrated = np.zeros(len(energies))
    
    for i_event, (event_channel, event_energy) in enumerate(zip(channels, energies)):
        coeffs = cal_params[cal_params['ch'] == event_channel]
        a, b, c = coeffs['a'][0], coeffs['b'][0], coeffs['c'][0]
        
        energies_calibrated[i_event] = a * event_energy**2 + b * event_energy + c
    
    return energies_calibrated

data_np["Energy_cal"] = energy_calibration(data_np["Channel"], data_np["Energy"], cal_params)

#%%

# =============================================================================
# Look for coincidences
# =============================================================================


# new_event_chunk2 = np.array(new_event_chunk, dtype='uint16,uint16,float64,float64,int64')

# Method 1: Using a list of tuples for dtype
# new_event_chunk2 = np.array(new_event_chunk, dtype=[
#     ('f0', 'float64'), ('f1', 'uint16'), ('f2', 'float64'),
#     ('f3', 'float64'), ('f4', 'int64')
# ])


#%%

output = {'test' : new_event_chunk2[:, 0].astype(np.uint16)}

with uproot.recreate("output\\output.root") as file:
    
    file["Data_C"] = output

