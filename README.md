# MNIST uncertainty

Create a micromamba env, activate it, install the packages.

```
export MAMBA_ROOT_PREFIX=/path/to/where/you/want/mambastuff/stored
eval "$(micromamba shell hook --shell=bash)"
micromamba create -n mymamba
micromamba activate mymamba
micromamba install -c conda-forge -n mymamba pytorch
```

# Getting started

* Install the required MNIST data.
* Try running tests/data.py to test that you have pytorch and can load up an image.

