Instalation
===============

Instalation of `rats` itself is simple pip installable.
>>> pip install rats

However, few modules need to be adjusted to work properly. These typically depend on external libraries, needed proper pathing to the directories. Without this, some functionallity of `rats` will break.

LDCU
============
`LDCU <https://github.com/delinea/LDCU>`_ is code for computing limb-darkening coefficients by Adrien Deline. This is used together with `batman package <https://github.com/lkreidberg/batman>`_ to compute light curves, which are mostly used for weighting of spectra. 
The list of adjustment before using this code:
1. Main directory of the `LDCU` project. This needs to be setup in <WIP: Fix the pathing such that it is setup in setup> 
2. Response functions adjustment <WIP: Simplify this. For spectra, uniform function within the wavelength range is most suitable>



petitRADTRANS
=====================
<WIP: Need to debug pRT with new version, as it has been recently updated to 3.0>


molecfit (esorex)
==================
`molecfit` is a code for telluric correction accessible from ESO website. The newer versions (>3.0) are fully implemented in the `esorex` / `esoreflex` workspace, which is assumed by `rats`. The standalone version, while in principle usable, is not considered by `rats`. 

Steps for fully functional installation:
0. Ensure the esorex/ esoreflex instalation is working on the provided test data. Please refer to ESO website for support with this step.
1. esorex path to bin file.
2. For instruments that are not supported by `rats`, a rc configuration file is necessary. For `ESPRESSO` and `HARPS`, these are automatically provided by `rats` within the FOLDERS_molecfit folder. As the rc file are UT sensitive, the `ESPRESSO` folders are separated based on the UT keyword. Finally, `NIRPS` folder is also provided, although it is not thoroughly tested.

StarRotator
======================
`StarRotator` is code for RM + CLV modeling of the spectra during the transit. 
<WIP: Setup the pathing in setup file>

RASSINE
========================
`RASSINE` is a code for normalizing spectra. It uses alpha-shapes strategy, which allows this code to also correct spectra for wiggle patterns.
<WIP: TBD>







