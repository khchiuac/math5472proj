# MATH 5472 Project

Code for report of "A Fast Algorithm for Maximum Likelihood Estimation of Mixture Proportions Using Sequential Quadratic Programming".



## Installation

Please follow the instructions in `code/requirements.txt`, the environment can be installed through `conda create --name <env> --file code/requirements.txt`.

Note that for `mosek`, a license would be required in an environment-specific path, for linux system it is `$HOME/mosek/mosek.lic`.  Please check [this link](https://www.mosek.com/products/academic-licenses/) for the instructions given by the official website.



## Experiment

Simply run `gmm.py` would give the solver's result. Note that it may take some time as no further optimization to the code has been done.

For plots can simply modify `plot.pt`.



## Reproducibility

Random seed is set to 0 for this project.