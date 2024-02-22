# RATS - Revealing Atmospheres with Transmission Spectra
 - This is a code for analysis of high-resolution transmission spectra.
 - The goal of this pipeline is to automatize most of the tasks when reducing the data without user input, making it possible to quickly setup multiple datasets.
 - This is still work in progress.
 - 


 - Usage:
    - `Template_transmission_spectroscopy.py` shows a typical pipeline to reduce `HARPS` and `ESPRESSO` data of a transit.
    - 

## molecfit
 - Before use:
   - Few adjustment for pathing is needed for `run_molecfit_all`. In particular, connection to the `esorex` recipe needs to be properly established.

## petitRADTrans
 - Before use:
    - Few adjustment for pathing needs to be done for `petitRADtrans` to be usable
    - The main importing one is location of high-resolution line lists, through the `OPACITY_LIST_LOCATION` variable in the `rats.modeling_CCF.py` file


## To be done:
 - Upper-limits calculation for non-detection
 - Detection functions (Fitting, significance calculations)
 - CCF functions
 - RM + CLV simulation (for now `StarRotator` can be used)
 - Clean up of plots functions
