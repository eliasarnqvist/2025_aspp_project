# 2025_aspp_project
Advanced scientific programming with python, repository for project

## Introduction

In the field of gamma spectrometry, the energies of gamma-rays originating in radioactive decay are measured and studied. Recent developments in data collection devices (digitizers) has led to the ability to record data from several detectors simultaneuously. For each detection event, the following is stored: 

- **Channel**: the input channel on the digitizer. Each channel corresponds to a unique detector. 
- **Energy**: the energy of the detected gamma-ray. This is recorded in arbitrary units that must be calibrated to energy units (keV) using a polynomial function. More on this later. 
- **Timestamp**: the time of detection in units of picoseconds. 
- **Flags**: information about digitizer saturation, pulse-pile-up, and similar special cases. This information can be used to filter out certain problematic events. 

This data is stored in list-mode in a ``.root`` file ([root webpage](https://root.cern/)). 

Coincidence detection events are where two or more events occur (nearly) at the same time. Such events are useful when performing some types of measurements. To obtain these events, the recorded data must be processed. In the end, for a coincidence event between two detectors (a and b), the following information should be saved: 

- **Channel a**: the channel of the first event. 
- **Channel b**: the channel of the second event. 
- **Energy a**: the energy of the first event. 
- **Energy b**: the energy of the second event. 
- **Time difference**: the time difference between the two events. 

### Project goals

The objective of this project is read, process, and save large data files efficiently using Python. In particular, the aim is to perform an energy calibration, identify coincidence events, and save data from the coincidence event to another file as quickly as possible. In the end, the data should also be plotted and explained briefly for some context. 

## Code explanation

Will add later...
