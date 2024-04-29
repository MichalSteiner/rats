.. OT documentation master file, created by
sphinx-quickstart on Wed Jan 17 12:59:34 2024.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

System parameters class
==============================
`RATS` module for handling system parameters. System parameters
are saved in a class system, several of which exists. The typical one is the `SystemParametersComposite` class, which
contains the information of planet, star and system.

Each variable is kept as a `astropy.nddata.NDDataArray` object. This includes the `parameter` value, `uncertainty`
attributes, while reference and parameter name are kept in the `meta` dictionary attribute.

The class is divided between the parameters of Planet (accessible by `.Planet.parameter` notation), Star (accessible
by `.Star.parameter` notation), System (accessible by `.System.parameter` notation) and Ephemeris (accessible by `
.Ephemeris.parameter` notation).

Currently, the simplest form of loading system parameters is through TAP service of NASA Exoplanet archive. This can
be done by: ::

    system_parameters = para.SystemParametersComposite(
        filename=save_directory + '/system_parameters.pkl'
        )
    system_parameters.load_NASA_CompositeTable_values(
        planet_name='WASP-31 b',
        force_load=True
        )

The first line here initiates the class. It generates all system parameters with None values. The filename argument
is used for saving the values in a pickle file, to avoid reloading the NASA TAP service. This save is only including
the TAP service values, values calculated through class methods or manual adding won't be saved.
The second line loads the values from NASA Exoplanet Composite Table, which takes as input the planet name. The
`force_load` keyword reloads the TAP service, even if the service was loaded recently. By default, every
week the TAP service is being updated. Meaning, if you run this line multiple time in one day without the
`force_load` option, only the first time it will actually do the TAP service extraction.


.. important::

    The NASA Exoplanet Composite Table does not necessarily provide most precise parameters. Always verify the
parameters provided with scientific literature. Furthermore, if analyzing old data, a less recent ephemeris might be
more precise due to closer mid transit value (the uncertainty scales with number of orbits passed since the mid
transit).

.. important::

    Please always cite all sources that have been used. For system parameter class, a method for generating LaTeX
table will be available soon [WIP].

.. important::
    Currently, no class for all exoplanets system is available. A similar class will be developed for plotting
diagrams of exoplanets population eventually, but this is currently low priority. Only the full


.. automodapi:: rats.parameters

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
