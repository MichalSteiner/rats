import rats.plots.plot_spectra_new as ps
import specutils as sp
import rats.spectra_manipulation as sm
import matplotlib.pyplot as plt

def molecfit_correction(data_corrected: sp.SpectrumList,
                        telluric_profiles: sp.SpectrumList,
                        data_uncorrected: sp.SpectrumList) -> [plt.Figure, plt.Axes]:
    """
    Testing the level of molecfit correction.

    Parameters
    ----------
    data_corrected : sp.SpectrumList
        Molecfit corrected data.
    telluric_profiles : sp.SpectrumList
        Telluric profiles used to correct for the spectra.
    data_uncorrected : sp.SpectrumList
        Raw spectra as kept by run_molecfit_all module.

    Returns
    -------
    fig : plt.Figure
        Figure object with the given test plot.
    axs:
        Artists objects with the molecfit correction plots.
    """
    master_corrected = sm.calculate_master_list(
        data_corrected,
        method= 'addition',
        )
    master_profiles = sm.calculate_master_list(
        telluric_profiles,
        method= 'average'
        )
    master_uncorrected = sm.calculate_master_list(
        data_uncorrected,
        method= 'addition',
        )
    
    number_of_nights = len(master_corrected)
    fig, axs = plt.subplots(number_of_nights-1, sharex=True)
    
    for ni in range(number_of_nights-1):
        axs[ni].plot(
            master_corrected[ni+1].spectral_axis,
            master_corrected[ni+1].flux,
            color='darkgreen',
            alpha= 0.6
            )
        axs[ni].plot(
            master_uncorrected[ni+1].spectral_axis,
            master_uncorrected[ni+1].flux,
            color='darkblue',
            alpha=0.4
            )
        
        ax2 = axs[ni].twinx()
        ax2.set_navigate(False)
        ax2.plot(master_profiles[ni+1].spectral_axis,
                 master_profiles[ni+1].flux,
                 color= 'darkred',
                 alpha=0.4
                 )
        axs[ni].plot(
            [],
            [],
            color='darkred',
            alpha=0.4
            )
        
        axs[ni].legend(['Telluric corrected',
                        'Raw spectra',
                        'Telluric profile'])
        
        axs[ni].set_xlabel(f'Wavelength [{master_profiles[ni+1].spectral_axis.unit:latex}]')
        axs[ni].set_ylabel(f'Flux [{master_corrected[ni+1].flux.unit:latex}]')
        
        ax2.set_ylabel(f'Telluric profile [{master_profiles[ni+1].flux.unit:latex}]')
        axs[ni]
        
        axs[ni].set_title('Telluric correction')
        ax2.set_ylim(-0.4,1.6)
    return fig, axs 


def normalization(pre_normalization: sp.SpectrumList,
                  post_normalization: sp.SpectrumList) -> [plt.Figure, plt.Axes]:
    # TODO - switch between spectra
    ...

def cosmic_correction() -> [plt.Figure, plt.Axes]:
    ...

def shifting_function() -> [plt.Figure, plt.Axes]:
    ...

def master_calculation(spectrum_list: sp.SpectrumList) -> [plt.Figure, plt.Axes]:
    ...

def RM_correction():
    ...
    