python ../graph_classification/train/main_graph_ogb.py --dataset ogbg-molbbbp --gnn_type GIN_Moe --emb_dim 300 --num_layers 4 --k 3 --seed 42 --save_dir results_gin_moe --device cuda:0 --min_layers 2 --epochs 120 --coef 0.001