.. OT documentation master file, created by
   sphinx-quickstart on Wed Jan 17 12:59:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Revealing Atmospheres with Transmission Spectroscopy
======================================================
Welcome to `RATS` (Revealing Atmospheres with Transmission Spectroscopy) package. This package allows extraction of high-resolution transmission spectroscopy, in particular from fiber-fed spectrographs like ESPRESSO and HARPS (although modification for other packages is mostly trivial). While currently the main focus is on high-resolution transmission spectroscopy, the Rossiter-McLaughlin effect analysis will be added soon. Furthermore, emission spectroscopy at high-resolution can also in principle be performed with minor modifications, though it has been completely untested.

.. important::

    This code is still under development and bugs might be present. In case you encounter some bug, please send the description to Michal.Steiner@unige.ch or create a new issue on GitHub. You can also use either of these to request a feature.

.. important::

    If you use this code, please cite Steiner et al. (2023). In future, more up-to-date reference will be available. Some modules also require additional external packages, please cite these as well. More specific instructions will be added in future.


This pipeline serves two purposes:

1. It is fully implemented transmission spectrocopy pipeline (soon Rossiter-McLaughlin will be implemented as well), which can be used "out-of-the-box", in particular with high-resolution spectrographs like HARPS and ESPRESSO.

2. It is a library of commonly used functions, utilizing the **specutils** package, from which user can use selected methods without the need to significantly modify their codes.


Get Started
===========

There are currently two template pipelines for `Transmission spectroscopy` and `Rossiter-McLaughlin` analysis. Its usage is provided in <WIP: Add template notebooks> :ref:`Get Started` notebooks. A script to initiate project might be provided eventually, but in short:
1. Create a `main_directory` (can be whatever empty folder). This is where all your analysis will be done in.
2. Create a `code` directory (the directory name can be different, but there are some reserved words).
3. Run `rats.single_use.setup_routine` function (check documentation for information on how to use). This function will create a tree directory and move the downloaded spectra into their respective directories.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
