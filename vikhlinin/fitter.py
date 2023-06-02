"""
This module provides a class for estimating the hydrostatic equilibrium of a galaxy 
cluster, given its density profile. It uses a variant of the Vikhlinin model to 
fit the profile.
"""

import unyt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize.optimize import OptimizeResult
from typing import List, Tuple, Optional
import warnings

starting_parameters_default = [3.e-3, 0.1, 0.6, 0.5, 0.4, 1.2]
parameter_bounds_default = [(0., np.inf),
                            (0., np.inf),
                            (0., np.inf),
                            (0., np.inf),
                            (0., np.inf),
                            (0., 5.)]

parameter_bounds_macsis = [(9.e-4, 9.e-3),
                            (0.01, 0.18),
                            (0.5, 0.75),
                            (1.49999, 1.500001),
                            (0.3, 0.6),
                            (2.0, 3.)]


def vikhlinin_density_model(radius: unyt.unyt_array, n_0: float, r_core: float, r_scale: float, 
                            alpha: float, beta: float, epsilon: float, yield_log: bool = True) -> unyt.unyt_array:
    """
    This function implements the Vikhlinin density model. It splits the 
    equation in a normalisation part and 3 additional terms, to simplify
    the notation.

    Parameters:
    radius: array of radii
    n_0, r_core, r_scale, alpha, beta, epsilon: Vikhlinin model parameters  
    """    
    term1 = (radius / r_core) ** (-alpha / 2)
    term2 = (1 + (radius / r_core) ** 2) ** (3 * beta / 2 - alpha / 4)
    term3 = ((1 + (radius / r_scale) ** 3) ** (epsilon / 6))
    
    if yield_log:
        return np.log10(n_0 * term1 / term2 / term3)

    return n_0 * term1 / term2 / term3

class VikhlininProfile:
    """
    This class is used to estimate hydrostatic equilibrium from a density profile.
    """
    
    def __init__(self, radii: unyt.unyt_array, density_profile: unyt.unyt_array, 
                 start_params: Optional[List[float]] = starting_parameters_default,
                 param_bounds: Optional[List[Tuple[float]]] = parameter_bounds_default):
        """
        Initialize the class with the input radii, density profile and optional 
        starting parameters for the model.

        Parameters:
        radii: array of radii
        density_profile: density profile corresponding to the radii
        starting_parameters: initial guesses for the Vikhlinin model parameters
        """        
        self.radii = radii
        self.density_profile = density_profile
        self.starting_parameters = start_params            
        self.parameter_bounds = param_bounds
            
        self.run_hse_fit()

    @staticmethod
    def residuals_density(free_parameters: List[float], density_data: unyt.unyt_array, 
                          radius_data: unyt.unyt_array) -> float:
        """
        Calculates the residuals between the model and the data.

        Parameters:
        free_parameters: current parameter values
        density_data: observed density profile
        radius_data: radii corresponding to the density profile
        """
        density_model = vikhlinin_density_model(radius_data, *free_parameters)
        error = density_model - density_data
        return np.sum(error * error)

    def density_fit(self, x: unyt.unyt_array, y: unyt.unyt_array) -> OptimizeResult:
        """
        Fits the model to the data.

        Parameters:
        x: array of radii
        y: observed density profile
        """
        optimize_result = minimize(self.residuals_density, 
                                   self.starting_parameters, 
                                   args=(y, x), 
                                   method='L-BFGS-B',
                                   bounds=self.parameter_bounds,
                                   options={'maxiter': 1e4, 'ftol': 1e-15})
        
        if not optimize_result.success:
            warnings.warn("Fit optimization did not succeed. Try a different method or different options.", RuntimeWarning)
        
        return optimize_result

    def run_hse_fit(self) -> None:
        """
        Runs the fitting process and updates the object's attributes with the fit results.
        """
        self.density_fit_params = self.density_fit(self.radii, np.log10(self.density_profile))
        self.density_profile_hse = 10 ** vikhlinin_density_model(self.radii, *self.density_fit_params.x)
        
        # Allocate the parameters individually
        self.n_0, self.r_core, self.r_scale, self.alpha, self.beta, self.epsilon = self.density_fit_params.x
        self.n_0 = unyt.unyt_quantity(np.array(self.n_0, dtype=np.float32), self.density_profile.units)
        self.r_core = unyt.unyt_quantity(np.array(self.r_core, dtype=np.float32), self.radii.units)
        self.r_scale = unyt.unyt_quantity(np.array(self.r_scale, dtype=np.float32), self.radii.units)
        
        # Allocate results of the optimisation
        self.success = self.density_fit_params.success
        self.message  = self.density_fit_params.message.decode('ascii')
        self.n_iterations  = self.density_fit_params.nit 
        
    def print_fit_parameters(self) -> None:
        """
        Prints the optimal fit parameters.
        """
        print(f"Fit parameters:")
        print(f"\t- Normalisation: {self.n_0:.3f}")
        print(f"\t- Core radius: {self.r_core:.3f}")
        print(f"\t- Scale radius: {self.r_scale:.3f}")
        print(f"\t- alpha: {self.alpha:.3f}")
        print(f"\t- beta: {self.beta:.3f}")
        print(f"\t- epsilon: {self.epsilon:.3f}")
        print(f"Optimizer status:")
        print(f"\t- Success: {self.success}")
        print(f"\t- Message: {self.message:s}")
        print(f"\t- Iterations performed: {self.n_iterations:d}")
