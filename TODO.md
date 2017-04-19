# TODO

## Sparse regression problem

- make actual regression sparse
- find optimum by some criterion
- simulate arbitrary inconstant background rate by rejection sampling
- check assignment is not breaking things
- step influence kernel
- better omega init

## Interval censored problem

- quantify the IIR stuff


## general

- Stop trying to duck-type everywhere; use strongly typed kernels everywhere and create factories for certain problems
- purge all array casting and once again, enforce types in some convenience factories
- My API is already very close to Stan, but with some design flaws. I should switch to a python equivalent style such as Edward to facilitate simulation-based inference.
- the background kernels could benefit from being written more like the foreground ones with tau params set accordingly
- improve initial guess to not
  - require guesses for params that will not be fit
  - too-easily leak values from re-used kernels
- tolerance calcs need to be more principled for kappa and omega
- optimisations - could save a loglik calc using penalty
