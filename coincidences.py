import uproot
import numpy as np
from numba import jit, uint16, uint64, int64
import time

# =============================================================================
# Settings
# =============================================================================

data_file = "data/big.root"
tree_name = "Data_R"

# Maximum time difference in ps between two events to be called coincident
time_max = np.uint64(250 * 1e3)




# =============================================================================
# Function
# =============================================================================

@jit("int64[:,:](uint16[:], uint64[:], uint16[:])", nopython=True)
def process_data(channels, timestamps, energies):
    """
    
    Parameters
    ----------
    channels : TYPE
        DESCRIPTION.
    timestamps : TYPE
        DESCRIPTION.
    energies : TYPE
        DESCRIPTION.

    Returns
    -------
    coincidence_events_chunk : TYPE
        DESCRIPTION.
    """
    
    n_events = len(channels)
    
    coincidence_events_chunk = np.empty((n_events * 2, 5), dtype=np.int64)
    
    i_save = 0
    
    for i_event in range(n_events):
        
        event_timestamp = timestamps[i_event]
        event_channel = channels[i_event]
        
        time_edge = event_timestamp + time_max
        
        for i_other in range(i_event + 1, n_events):
            
            other_timestamp = timestamps[i_other]
            other_channel = channels[i_other]

            if other_timestamp < time_edge and other_channel != event_channel:
                # Coincidence has occurred
                
                event_energy = energies[i_event]
                other_energy = energies[i_other]
                
                time_difference = np.int64(other_timestamp - event_timestamp)
                
                # Save the coincidence event as well as its complement
                new_row = np.array([event_channel, other_channel, event_energy, other_energy, time_difference], dtype=np.int64)
                coincidence_events_chunk[i_save] = new_row
                i_save += 1
                new_row = np.array([other_channel, event_channel, other_energy, event_energy, -time_difference], dtype=np.int64)
                coincidence_events_chunk[i_save] = new_row
                i_save += 1
                
            else:
                # Avoid looking at unneccessary data
                break
    
    # Remove the empty part of our initial array
    coincidence_events_chunk = coincidence_events_chunk[:i_save]
    
    return coincidence_events_chunk

# =============================================================================
# Import data
# =============================================================================

buffer_size = "100 MB"

start_time = time.time()

illegal_flags = np.array([0x80, 0x400, 0x480], dtype=np.uint32)

legal_channels = np.array([1, 2, 3, 4], dtype=np.uint16)

legal_branches = ["Channel", "Energy", "Timestamp"]

events = uproot.open(data_file + ":" + tree_name)


total_entries = events.num_entries
entries_per_chunk = events.num_entries_for(memory_size=buffer_size)
estimated_chunks = int(total_entries/entries_per_chunk + 1)
print("Estimated number of chunks: " + str(estimated_chunks))
i_chunk = 0


inflation_size = 1e6
coincidence_events = np.empty((int(inflation_size), 5), dtype=np.int64)
filled_rows = 0
available_rows = len(coincidence_events)

for event_chunk, report in events.iterate(step_size=buffer_size, library="np", report=True):    
    # Apply selections
    selection = np.isin(event_chunk["Flags"], illegal_flags, invert=True)
    
    # Now event_chunk is a dict containing np arrays
    
    chunk_coincidences = process_data(event_chunk["Channel"], event_chunk["Timestamp"], event_chunk["Energy"])
    new_rows = len(chunk_coincidences)
    
    if filled_rows + new_rows > available_rows:
        print("...extending save array...")
        
        coincidence_events = np.vstack((coincidence_events, np.empty((int(inflation_size), 5), dtype=np.int64)))
        available_rows += int(inflation_size)
    
    coincidence_events[filled_rows:filled_rows+new_rows, :] = chunk_coincidences
    
    filled_rows += new_rows
    
    # Print progress
    i_chunk += 1
    # print(report)
    chunk_last_entry_number = report.stop
    progress = round(chunk_last_entry_number / total_entries * 100, 1)
    print("Finished chunk " + str(i_chunk) + ", progress: " + str(progress) + " %")



coincidence_events = coincidence_events[:filled_rows]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

#%%




output = {'test' : new_event_chunk2[:, 0].astype(np.uint16)}

with uproot.recreate("output\\output.root") as file:
    
    file["Data_C"] = output



