import argparse
import os.path as osp
import sys
sys.path.append('../')
import torch
from model_proteins.gnn import GCN_OGB
from model_proteins.gnn_moe import GCN_OGB_Moe
import numpy as np
import random
from logger import Logger
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model,optimizer,data,train_idx,use_moe = False):

    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    if use_moe:
        loss = criterion(out, data.y[train_idx].to(torch.float))+model.load_balance_loss
    else:
        loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-proteins')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--coef', type=float, default=0.0001)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42,
                        help='seed number')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='seed number')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gnn_type',type=str,default='GCN',
                        choices=['GCN','GCN_Moe'])
    parser.add_argument("--dropout", type=float, default=.5,
                        help="input feature dropout")
    parser.add_argument("--fin_mlp_layers", type=int, default=1,
                        help="final mlp layers")
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--gate_type', type=str, default='GCN',choices=['GCN','liner','SAGE'])
    parser.add_argument('--min_layers', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='results')

    args = parser.parse_args()
    logger = Logger(args.num_seeds,args.runs, args)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')
    dataset = PygNodePropPredDataset(
        name=args.dataset,root=path, transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    
    if args.gnn_type == 'GCN' or 'GCN_Moe':
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
    
    data = data.to(device)
    seed_lst = [i+args.seed for i in range(args.num_seeds)]

    for seed in seed_lst:
        set_random_seed(seed)
        for run in range(args.runs):
            if args.gnn_type == 'GCN':
                model = GCN_OGB(
                    in_channels=data.num_features, 
                    hidden_channels=args.hidden_channels,
                    out_channels= 112,
                    num_layers=args.num_layers, 
                    dropout=args.dropout).to(device)
            elif args.gnn_type == 'GCN_Moe':
                model = GCN_OGB_Moe(
                    gnn_type='GCN',
                    gate_type=args.gate_type,
                    in_channels=data.num_features,
                    hidden_channels=args.hidden_channels, 
                    out_channels=112,
                    num_layers=args.num_layers,
                    min_layers=args.min_layers,
                    topK=args.k,
                    coef=args.coef,
                    dropout= args.dropout
                ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            evaluator = Evaluator(name=args.dataset)
            for epoch in range(1, 1 + args.epochs):
                if 'Moe' in args.gnn_type:
                    loss = train(model,optimizer,data,train_idx,True)
                else:
                    loss = train(model,optimizer,data,train_idx)
                train_acc,val_acc,test_acc =  test(model, data, split_idx, evaluator)
                result = (train_acc, val_acc, test_acc)
                logger.add_result(seed-args.seed,run, result)

                if epoch % args.log_steps == 0:
                    train_acc, val_acc, test_acc = result
                    print(f'Seed: {seed:02d}, '
                        f'Epoch: {epoch:03d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.4f}%, '
                        f'Valid: {100 * val_acc:.4f}%, '
                        f'Test: {100 * test_acc:.4f}%')
            logger.print_statistics(args.dataset,'classification',args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.dropout,args.coef,args.k,seed-args.seed,run)
        logger.print_statistics(args.dataset,'classification',args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.dropout,args.coef,args.k,seed-args.seed)
    logger.print_statistics(args.dataset,'classification',args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.dropout,args.coef,args.k,)

if __name__ == "__main__":
    main()