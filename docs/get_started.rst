Get Started
===============
In this section we will look at how to use `rats` pipeline for transmission spectroscopy and Rossiter-McLaughlin analysis.


Transmission spectroscopy
===============================

Theory in a nutshell: Transmission spectroscopy is a method to analyse additional absorption features caused by a medium between source of light and observer. In the context of exoplanet, the source of light is star, the observer is a detector on a telescope and the medium is the exoplanet atmosphere. This is because the exoplanet itself is opaque, and does not modify the stellar spectrum, only removes a fraction of light.

In practice, the situation is more complicated, and multiple corrections are necessary.

Step 0: Download data from DACE:
Step 0.1: Generate a directory tree for the project, using the `rats.single_use` modules

Step 1: Load data with `rats.load` modules:
Step 1.1: In case of ESO instruments, the supported instruments are: `HARPS` and `ESPRESSO`. In principle, `NIRPS` can be also easily used, as the DRS behind is the same. Other instruments require writing of this module for them. 


Step 2: Telluric correction with `molecfit`:
Step 2.1: `rats` has a wrapper for running molecfit on the background through the pipeline in `rats.run_molecfit_all`. This wrapper allows correcting for of all input spectra with prepared templates located in the `FOLDERS_molecfit` folder of rats. These templates are almost finished for supported instruments, with only the wavelength regions being the main thing to change. This can be done through interactive plot interface, which allows selection and deselection of wavelength windows, which is automatically saved in `WAVE_INCLUDE`, `WAVE_EXCLUDE` and `PIXEL_EXCLUDE` <WIP: `PIXEL_EXCLUDE` is not working currently>. This file is then refered to by the `molecfit_*.rc` files, meaning no additional adjustment is necessary.
Step 2.2: The output of `rats.run_molecfit_all` can be loaded by `rats.load.molecfit`. 

.. important::
    The output of `molecfit` is different then the output of `rats.run_molecfit_all`. `rats` saves additional info about the telluric profile and the original spectrum used, which can be then plotted for cross-check.

Step 3: Load system parameters

Step 4: Rebinning to common wavelength grid:

.. important:: 
    This step is currently necessary, as `rats` is not always checking whether the spectrum is on the same wavelength grid. However, it makes it difficult to use multiple instruments at the same time. <WIP: Choose between separating function also based on instruments or allow binning to same wavelength grid for vastly different instruments>. 
    Known bug: Furthermore, in case instruments are not overlapping with wavelength range, this function will remove access to the second instrument completely.


