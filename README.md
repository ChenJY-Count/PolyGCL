## PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters

This repository contains a PyTorch implementation of "PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters".


## Environment Settings    
- pytorch 1.11.0
- numpy 1.20.3
- torch-geometric 1.7.2
- dgl-cu113 0.8.2
- scipy 1.7.1
- seaborn 0.11.2
- scikit-learn 0.24.2

### Datasets
We provide datasets in the folder 'data'.


## Reproduce the results

### On real-world datasets
You can run the following commands directly.

```sh
sh exp_PolyGCL.sh
```
Heterophilic datasets
```sh
cd HeterophilousGraph
sh exp_PolyGCL.sh
```


### On synthetic datasets

Generate the cSBM data firstly.
```sh
cd cSBM
sh create_cSBM.sh
```
Then run the following command directly.
```sh
sh run_cSBM.sh
```
