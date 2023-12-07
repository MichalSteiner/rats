# TODO:
"""
Main idea:
    Create a hexagonal grid to simulate the planet transit.
    Create fine sampled grid on the planetary track
    Each cell should be scaled by limb-darkening and shifted by local stellar velocity.
        How to do this?
    
    First order approximation creates a grid such that the planet is in center of a cell, and the next spectrum is in the center of the next cell.
        As such, the RM+CLV effect is full grid - the one occulted cell
    Second order approximation creates a fine grid where the position of the planet changes over the exposure. 
        The RM+CLV effect is than a combination of cell subtractions, normalized to planet area.
    Third order approximation simulates the entire exposure, with movement of the planet.
        The RM+CLV effect is than sampled more finely in time than the exposure.
"""

class ModelGrid:
    """
    Model grid for the star. Can simulate a transit event to estimate the RM+CLV effect.
    """
    pass

    def setup_stellar_grid():
        return
    
    def cut_non_transited_grid():
        return
    
    def scale_by_cross_section():
        # TODO - On the edge, cut the grid to circular one.
        return
    
    
    def simulate_transit():
        return
