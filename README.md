# CRPSnb: Neighborhood-Based Continuous Ranked Probability Score (CRPS)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)




<p>
This Python package provides an efficient implementation of the neighborhood-based Continuous Ranked Probability Score (CRPS), a skill score used to assess the performance of ensemble forecasts. It builds upon the traditional CRPS by accounting for the spatial structure of the forecasts and observations.
</p>

<p>
This implementation is based on:

- Stein, Joël, and Fabien Stoop. "Neighborhood-based ensemble evaluation using the CRPS." Monthly Weather Review 150, no. 8 (2022): 1901-1914.
</p>


## Key Features

- **Vectorized Implementation:** Efficient calculations for large datasets using NumPy vectorization.
- **Parallel Processing:** Option to parallelize computations for further speedup.

## Installation

**clone the repository**
```bash
git clone URL

cd crpsnb

pip install .
```

## Usage

```python
import numpy as np
from crpsnb.crps import crps_neighboor, crps_neighborhood_chunk, crps_neighborhood_chunk_nonvectorized

RSEED = 1915 # random seed for reproducibility of sample generation

ENS_SIZE = 5 # number of ensemble members for each time step
OBS_SIZE = 3 # number of observations for each time step
N_TSTEPS = 10 # number of time steps

drng = np.random.default_rng(RSEED)

ensemble = drng.normal(size=(N_TSTEPS, ENS_SIZE)).round(1)
yobs = drng.normal(size=(N_TSTEPS, OBS_SIZE)).round(1)

crps_nb = crps_neighboor(ensemble, yobs, chunk_size=2, njobs=2)

print(
    f"CRPS: {crps_nb}"
)
```

## Contact

Feel free to contact me for any questions or suggestions: alireza.amani101@gmail.com
