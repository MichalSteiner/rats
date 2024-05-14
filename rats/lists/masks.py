from dataclasses import dataclass
import specutils as sp


@dataclass
class Mask:
    instrument: str
    mask_regions: sp.SpectralRegion
    
    def mask_spectrum_list(self,
                           spectrum_list: sp.SpectrumList):
        ...
    

ESPRESSO_mask = Mask(
    instrument='ESPRESSO',
    mask_regions=sp.SpectralRegion(3500*u.AA, 3800*u.AA) + sp.SpectralRegion(7870*u.AA, 8000*u.AA)
)