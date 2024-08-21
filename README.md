# DA-MoE: Addressing Depth-Sensitivity in Graph-Level Analysis through Mixture of Experts
## Overview

This paper presents **DA-MoE**, a novel MoE framework dedicated to addressing depth-sensitivity issue in graph-structured data. DA-MoE utilized different GNN layers as experts and allowed each individual graph to adaptively select experts. Additionally, this framework highlights two key modifications: the structure-based gating network and balanced loss function. Furthermore, comprehensive experiments on the TU dataset and open graph benchmark (OGB) have shown that DA-MoE consistently surpasses existing baselines on various tasks, including graph, node, and link-level analyses.

### Python environment setup with Conda

```
conda create -n damoe python=3.11
conda activate damoe
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install numpy
pip install ogb
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Running DA-MoE

```
conda activate damoe
# Running DA-MoE to complete graph classification for TU dataset.
sh ./scripts/run_proteins.sh 
# Running DA-MoE to complete graph level task for OGB.
sh ./scripts/run_bbbp.sh
sh ./scripts/run_freesolv.sh  
# Running DA-MoE to complete node classification for OGB.
sh ./scripts/run_arxiv.sh
# Running DA-MoE to complete link prediction for OGB.
sh ./scripts/run_ddi.sh
```

Supported datasets:

- TU dataset: `NCI1`,  `PROTEINS`,  `MUTAG`,  `IMDB-BINARY`, `IMDB-MULTI`, `REDDIT-BINARY`, `COLLAB`
- OGB Graph Classification: `ogbg-molhiv`,  `ogbg-moltox21`,  `ogbg-moltoxcast`, `ogbg-molbbbp`
- OGB Graph Regression: `ogbg-molfreesolv`,  `ogbg-molesol`
- OGB Node Classification: `ogbn-arxiv`,  `ogbn-proteins`
- OGB Link Prediction: `ogbl-ppa`, `ogbl-ddi`

## Baselines

- GCN: https://github.com/tkipf/gcn
- GIN: https://github.com/weihua916/powerful-gnns
- GatedGCN: https://github.com/pyg-team/pytorch_geometric
- GMoE: https://github.com/VITA-Group/Graph-Mixture-of-Experts/tree/main

## Datasets

Datasets from Open Graph Benchmark will be downloaded automatically using OGB's API when running the code.

For TU dataset, we adopt the dataset split from GIN implementation and can be download at [here](https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip) , and unzip them to the **./graph_classification/datasets_split** directory.
