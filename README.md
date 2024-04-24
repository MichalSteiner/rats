# RATS - Revealing Atmospheres with Transmission Spectra

## Purpose:
 - This is a code for analysis of high-resolution transmission spectra. Currently, implemented instruments are `HARPS` and `ESPRESSO`. `NIRPS` DRS pipeline uses the same DRS as these two instruments, and while untested, the loading functions should work correctly.
   - Adaptation for other instrument is mainly about:
      1. Loading functions and formating spectra in `RATS` format.
      2. Instrument-specific corrections
      3. Checking the tree-directory setup works properly.
 - The goal of this pipeline is to automatize most of the tasks when reducing the data without user input, making it possible to quickly setup multiple datasets. This is done by generalized template for transmission spectroscopy (RM analysis to be implemented), which takes as input:
   - Filename location as downloaded from `DACE`. No other ways of downloading data have been tested. `RATS` as first step when opening data creates a organized tree directory with all the data, which is then assumed by the pipeline. 
 - This package is still work in progress.

 - Example usage:
    - `Template_transmission_spectroscopy.py` shows a typical pipeline to reduce transmission spectroscopy `HARPS` and `ESPRESSO` data of a transit.
    - `Template_rossiter_mclaughlin_effect.py` shows a typical pipeline to reduce RM effect using `HARPS` and `ESPRESSO` spectrographs, using the "Revolutions" method (Bourrier et al. 2021)
    - TODO: Create a script to run in terminal for automatic setup of these.
    
## Before use:
  - Few modules are depending on external libraries, that need further setup. Generally, this includes filepaths to the external libraries.
  - TODO: The setup will be moved to singular file

### molecfit:
 - Before use:
   - Few adjustment for pathing is needed for `run_molecfit_all`. In particular, connection to the `esorex` recipe needs to be properly established.
   - The rest of the pathing should be setup automatically. 

### petitRADTrans:
 - Before use:
    - Few adjustment for pathing needs to be done for `petitRADtrans` to be usable
    - The main importing one is location of high-resolution line lists, through the `OPACITY_LIST_LOCATION` variable in the `rats.modeling_CCF.py` file.

### LDCU:
 - Before use:
  - A filepath to the code main directory needs to be provided.

### StarRotator:
  - Before use:
    - A filepath to main directory of the code needs to be provided.
    - TODO: Implement the code so StarRotator is loaded as external library, instead of being within the RATS package

## To be done:
 - Upper-limits calculation for non-detection
 - Detection functions (Fitting, significance calculations)
 - CCF functions
 - RM + CLV simulation (for now `StarRotator` can be used)
 - Clean up of plots functions
 - RM Revolutions technique for characterization of RM effect

## Feedback: 
 - Please provide any feedback to Michal Steiner (Michal.Steiner@unige.ch) or through the issues interface on GitHub.
