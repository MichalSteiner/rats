Parameters:
=====================================
This module main purpose is to retrieve system parameters. It allows to load parameters from NASA archive through the TAP servive. Furtermore, many convenience methods are added to the class, so the various classes are not just parameters containers. As an example, we extract stellar model from te saved stellar parameters, calculate parameters which were not provided by NASA archive (as it doesn't allow for derived parameters calculation).

.. important::

    Parameters are saved as `NDDataRef` arrays from `astropy.nddata` package. These allow to handle units, uncertainties and meta parameters at the same time.

.. important::

    Many features are to be implemented here. First, a parameters class container for full sample of planets. Second, various plots, such as highlighting exoplanet within the population.

.. important::

    When using parameters loaded from NASA archive, please always double-check the values are correct. Sometimes, the parameters are incorrect, or not the most precise available.

.. automodapi:: parameters
   :no-inheritance-diagram: