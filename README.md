# 2025_aspp_project
Advanced scientific programming with python, repository for project

## Introduction

In the field of gamma spectrometry, the energies of gamma-rays originating in radioactive decay are measured and studied. Recent developments in data collection devices has led to the ability to record data from several detectors simultaneuously. The device recording data is called a digitizer and the recorded data consists of: 

- **Channel**: the input channel on the digitizer. Each channel corresponds to a unique detector. 
- **Energy**: the energy of the detected gamma-ray. Recorded in arbitrary units that must be calibrated to energy using a polynomial function. 
- **Timestamp**: the time of detection in units of picoseconds. 
- **Flags**: information about digitizer saturation, pulse-pile-up, and other data that should be excluded. 

