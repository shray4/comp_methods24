<p align="center">
    <img width="500", src="logo2.png">
    <!-- put our logo here instead of google -->
</p>

<h2 align="center">
    <b>M</b>onte <b>c</b>arlo <b>F</b>or <b>A</b>GN <b>C</b>hannel <b>T</b>esting and <b>S</b>imulations
    <br>
    <!-- <a href="https://github.com/TeamLEGWORK/LEGWORK-paper">
        <img src="https://img.shields.io/badge/release paper-repo-blue.svg?style=flat&logo=GitHub" alt="Read the article"/>
    </a>
    <a href="https://codecov.io/gh/TeamLEGWORK/LEGWORK">
        <img src="https://codecov.io/gh/TeamLEGWORK/LEGWORK/branch/main/graph/badge.svg?token=FUG4RFYCWX"/>
    </a>
    <a href='https://legwork.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/legwork/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://ascl.net/2111.007">
        <img src="https://img.shields.io/badge/ascl-2111.007-blue.svg?colorB=262255" alt="ascl:2111.007" />
    </a>
    <a href="mailto:tomjwagg@gmail.com?cc=kbreivik@flatironinstitute.org">
        <img src="https://img.shields.io/badge/contact-authors-blueviolet.svg?style=flat" alt="Email the authors"/>
    </a> -->
</h2>

<p align="center">
    A python package that does the AGN channel for you!    And now with kick velocities by Shawn Ray!
</p>

### Installation

The latest development version is available directly from my [GitHub Repo](https://github.com/shray4/comp_methods24). To start, clone the repository onto your machine:

```
    git clone https://github.com/shray4/comp_methods24
    cd comp_methods24
```
Next, we recommend that you create a Conda environment for working with McFACTS.
You can do this by running

```
    conda create --name mcfacts-dev "python>=3.12.0" pip "numpy>=1.23.1" "scipy>=1.11.2" "matplotlib>=3.5.2" -c conda-forge -c defaults

```

And then activate the environment by running

```
    conda activate mcfacts-dev
```

And then install the dependency packages for McFACTS by running
```
    pip install .
```
After this, you'll want to install the dependency packages for evolve_binary
<p align="center">
!!!!!!!!!!!  evolve_binary is a computational method developed by [GitHub Repo](https://github.com/keefemitman)  !!!!!!!!!!!
</p>

Evolve a binary through merger using PN equations and a NR remnant surrogate!

Required Installation Process:
- install Julia via `curl -fsSL https://install.julialang.org | sh`
- create a mcfacts-dev conda environment via `make setup` in your McFACTS repository
- install `sxs` via `conda install -c conda-forge sxs numba::numba numba::llvmlite`
- `conda install joblib==1.4.2 quaternionic==1.0.13`
- `pip install juliacall==0.9.20 scikit_learn==1.5.2 sxs==2024.0.22 uncertainties==3.2.2`
- - download `surrogate.joblib` from [this Google Drive](https://www.dropbox.com/scl/fo/p33rqfjew5vu5qzksu32w/AEr4moWujITfl46ezybjE1Q?rlkey=1lladw82d8twlpt2xi5hidscv&st=xctpnkyj&dl=0) and place in the following directory `src/mcfacts/external/evolve_binary/`


Now all that's left to do is run McFACTS with the new kick velocity model!

```
    make plots
    python removing_hash.py
    cd src/mcfacts/external/evolve_binary
    python evolve_mcfacts.py
```
The outputs you should expect to generate can be seen in `src/mcfacts/external/evolve_binary/evolve_mcfacts.ipynb` which can also be run in case of issues with installation and running of main python file.
