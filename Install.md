This project has been tested with Ubuntu 18.04.

```
# create a conda environment and install requirements

conda create -n ambi python=3.7
conda activate ambi
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
conda install -c anaconda numpy
conda install -c anaconda scipy 
conda install -c conda-forge matplotlib
conda install -c conda-forge musdb
conda install -c conda-forge librosa
pip install git+https://github.com/SiggiGue/pyfilterbank.git    

```
With this newly created environment, you can start using the repo.
