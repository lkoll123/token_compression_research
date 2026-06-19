#Setup

##1. Conda Env

Setup Conda Environment via:

```bash
conda env create -f environment.yml
conda activate fna_torch3d
```

##2. Git Submodules

Various experimental models have been included as git submodules. Initialize via: 

```bash
git submodule update --init --recursive
```
