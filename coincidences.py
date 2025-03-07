# Elias Arnqvist, 2025

import uproot
import numpy as np
from numba import jit
import time

# =============================================================================
# Settings
# =============================================================================

# Properties of the input file
input_file = "data/big.root"
input_tree = "Data_R"

# Big input files have to be read in chunks of specified size
chunk_size = "100 MB"

# Maximum time difference in ps between two events to be called coincident
time_difference_max = np.uint64(250 * 1e3)

# Choose how to save the coincident data
output_file = "output/coincidences_1.root"
output_tree = "Data_C"

# Used to filter out flags in input data corresponding to saturation events
illegal_flags = np.array([0x80, 0x400, 0x480], dtype=np.uint32)

# =============================================================================
# Coincidence extractor
# =============================================================================

# Main function that looks for coincidence events
@jit("int64[:,:](uint16[:], uint64[:], uint16[:], uint32[:])", nopython=True)
def process_data(channels, timestamps, energies, flags):
    """
    Processes list-mode detector data to find and extract coincidence events. 
    
    Parameters
    ----------
    channels : numpy.ndarray
        1D array of unsigned 16-bit integers representing event channels
    timestamps : numpy.ndarray
        1D array of unsigned 64-bit integers representing event timestamps
    energies : numpy.ndarray
        1D array of unsigned 16-bit integers representing event energy
    flags : numpy.ndarray
        1D array of unsigned 32-bit integers representing event flags

    Returns
    -------
    coincidence_events_chunk : numpy.ndarray
        2D array of signed 64-bit integers containing coincidence events
    """
    
    # Number of events in data being processed
    n_events = len(channels)
    # Inflate an empty array to store coincidence events in
    coincidence_events_chunk = np.empty((n_events * 2, 5), dtype=np.int64)
    # Number of rows saved in the coincidence event array
    i_saved_rows = 0
    
    # Loop over all events in the list-mode data
    for i_event in range(n_events):
        
        event_timestamp = timestamps[i_event]
        event_channel = channels[i_event]
        event_flag = flags[i_event]
        
        # The largest possible timestamp to still be in coincidence
        timestamp_edge = event_timestamp + time_difference_max
        
        # Loop through all other events after the first one
        for i_other in range(i_event + 1, n_events):
            
            other_timestamp = timestamps[i_other]
            other_channel = channels[i_other]
            other_flag = flags[i_other]
            
            # Check if a coincidence event has occurred
            cond1 = other_timestamp < timestamp_edge
            cond2 = other_channel != event_channel
            cond3 = event_flag not in illegal_flags
            cond4 = other_flag not in illegal_flags
            if cond1 and cond2 and cond3 and cond4:
                
                event_energy = energies[i_event]
                other_energy = energies[i_other]
                
                # The time difference between the two coincidence events
                time_difference = np.int64(other_timestamp - event_timestamp)
                
                # Save the coincidence event to our array from before
                new_row = np.array([event_channel, other_channel, 
                                    event_energy, other_energy, 
                                    time_difference], dtype=np.int64)
                coincidence_events_chunk[i_saved_rows] = new_row
                i_saved_rows += 1
                # Also save the complementary event
                new_row = np.array([other_channel, event_channel, 
                                    other_energy, event_energy, 
                                    -time_difference], dtype=np.int64)
                coincidence_events_chunk[i_saved_rows] = new_row
                i_saved_rows += 1
            else:
                # Here events are too late to be in coincidence
                # Break to avoid looking at unneccessary data
                break
    
    # Remove the still empty part of our initial array
    coincidence_events_chunk = coincidence_events_chunk[:i_saved_rows]
    
    return coincidence_events_chunk

# Start timer
start_time = time.time()

# The list-mode data tree
print("Opening root file: " + input_file)
print("\tTree name: " + input_tree)
events = uproot.open(input_file + ":" + input_tree)

# Some information used to estimate progress
total_entries = events.num_entries
entries_per_chunk = events.num_entries_for(memory_size=chunk_size)
estimated_chunks = int(total_entries/entries_per_chunk + 1)
print("Estimated number of chunks to read: " + str(estimated_chunks))
i_chunk = 0

# Inflate an empty array to store all coincidence events in
inflation_size = 1e6
coincidence_events = np.empty((int(inflation_size), 5), dtype=np.int64)
filled_rows = 0
available_rows = len(coincidence_events)

# Open the list-mode data tree chunk by chunk
for event_chunk, report in events.iterate(step_size=chunk_size, library="np", report=True):
    # Now event_chunk is a dict containing np arrays
    
    # Call the processing function with the chunk data
    chunk_coincidences = process_data(event_chunk["Channel"], 
                                      event_chunk["Timestamp"], 
                                      event_chunk["Energy"], 
                                      event_chunk["Flags"])
    
    # Keep track of rows we add to the coincidence data array
    new_rows = len(chunk_coincidences)
    if filled_rows + new_rows > available_rows:
        print("\t\t...extending save array...")
        coincidence_events = np.vstack((coincidence_events, np.empty((int(inflation_size), 5), dtype=np.int64)))
        available_rows += int(inflation_size)
    # Add the chunk coincidence data to the array from before
    coincidence_events[filled_rows:filled_rows+new_rows, :] = chunk_coincidences
    filled_rows += new_rows
    
    # Print progress
    i_chunk += 1
    chunk_last_entry_number = report.stop
    progress = round(chunk_last_entry_number / total_entries * 100, 1)
    print("\tFinished chunk " + str(i_chunk) + ", progress: " + str(progress) + " %")

# Remove empty rows in the array
coincidence_events = coincidence_events[:filled_rows]

# Print the time it took
end_time = time.time()
elapsed_time = end_time - start_time
print("Coincidence processing time: " + str(round(elapsed_time, 1)) + " seconds")

# =============================================================================
# Save coincidence data
# =============================================================================

# Name of the branches to make in the output root file
coincidence_columns = ["Channel_a", "Channel_b", "Energy_a", "Energy_b", "Time_difference"]
# Prepare dictionary with data for the root file
output = {key : coincidence_events[:, i] for i, key in enumerate(coincidence_columns)}

# Write data to a root file
with uproot.recreate(output_file) as file:
    file[output_tree] = output

print("Data saved to root file: " + output_file)
print("\tTree name: " + output_tree)

# The end
