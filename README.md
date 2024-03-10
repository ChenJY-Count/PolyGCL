## PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters

This repository contains a PyTorch implementation of ICLR 2024 paper "[*PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters*](https://openreview.net/pdf?id=y21ZO6M86t)".


## Environment Settings    
- pytorch 1.11.0
- numpy 1.20.3
- torch-geometric 1.7.2
- dgl-cu113 0.8.2
- scipy 1.7.1
- seaborn 0.11.2
- scikit-learn 0.24.2

### Datasets
We provide the small datasets in the folder 'data'. You can access the heterophilic datasets and the large heterophilic graph arXiv-year via [heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs) and [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) respectively.


## Reproduce the results

### On real-world datasets
You can run the following commands directly.

```sh
sh exp_PolyGCL.sh
```
Heterophilic datasets:
```sh
cd HeterophilousGraph
sh exp_PolyGCL.sh
```
Large heterophilic graph arXiv-year:
```sh
cd non-homophilous
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

## Acknowledgements 
This project includes code or ideas inspired by the following repositories:
 - [ChebNetII](https://github.com/ivam-he/ChebNetII)
 -  [MVGRL](https://github.com/kavehhassani/mvgrl)
 - [DGI](https://github.com/PetarV-/DGI)

## Contact
If you have any questions, please feel free to contact me with [jy.chen@ruc.edu.cn](mailto:jy.chen@ruc.edu.cn).
