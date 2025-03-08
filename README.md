# 2025_aspp_project
Advanced scientific programming with python, repository for project

**Status of project:** finished

## Introduction

In the field of gamma spectrometry, the energies of gamma-rays originating in radioactive decays are measured and studied. Recent developments in data collection devices (digitizers) has led to the ability to record data from several detectors simultaneuously. For each detection event, the following is recorded: 

- **Channel**: the input channel on the digitizer. Each channel corresponds to a unique detector. 
- **Energy**: the energy of the detected gamma-ray. It is in arbitrary units that need to be calibrated to energy (keV). 
- **Timestamp**: the time of detection in units of picoseconds. 
- **Flags**: information about digitizer saturation, pulse-pile-up, and similar special cases. This information can be used to filter out problematic events. 

This data is stored in list-mode in a ``.root`` file, as is common in nuclear and particle physics applications ([CERN root webpage](https://root.cern/)). 

Coincidence detection events are where two or more events occur (nearly) at the same time. Such events are useful when performing some types of measurements. To obtain these events, the recorded data must be processed. In the end, for a coincidence event between two detectors (a and b), the following information should be saved: 

- **Channel a**: the channel of the first event. 
- **Channel b**: the channel of the second event. 
- **Energy a**: the energy of the first event. 
- **Energy b**: the energy of the second event. 
- **Time difference**: the time difference between the two events. 

### Project goals

The objective of this project is to read, process, and save list-mode detector data files efficiently using Python. In particular, the aim is to identify coincidence events and save these events to another file as quickly as possible. Moreover, large data files should be readable without any issues. For some context, the data should also be plotted in the end. 

## Running ``extract_coincidences.py``

The code can be run through the command ``python3 extract_coincidences.py``. The python file can be edited to change locations of input and output data. Included in this repository is a relatively small (~6 MB) input data file for testing, which my system processes in about 0.1 seconds. I have tried it with a larger (~1.3 GB) file as well, which my system processes in about 10 seconds. The big file is too large to be included in this repository. 

Two, perhaps, uncommon libraries are needed: ``numba`` and ``uproot``. A just-in-time compiler from ``numba`` is used. It compiles a function when the function is called the first time, greatly speeding up execution for following function calls. Using ``numba``, similar performance to ``cython`` is reached, but I think the ``numba`` code is easier to read and understand, as everything is in one Python file. To read ``.root`` files, ``uproot`` is needed. An alternative is using the official ``pyROOT``, but this is only (easily) useable in Linux. 

## Running ``plot_coincidences.py``

The code can be run through the command ``python3 plot_coincidences.py``. This calibrates the energy of the extracted coincident events to get units of keV (based on a calibration performed outside the scope of this work). Then, an exaple plot is created showing the energy-energy histogram of coincident events between channels 3 and 4. The plot is saved as well. 

## Conclusions

Python code was written that reads detector list-mode data and extrancts coincidence events from the data. These coincidence events are then saved in another file. The implemented code has been partially optimized to reach good performance and fast execution times. The goal of this project has therefore been reached. 
