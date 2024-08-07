import argparse
import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model.gnn import GIN,GCN,GatedGCN
from model.gnn_moe import GNN_Moe
from data import get_dataset
import numpy as np
import random
from logger import Logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model,optimizer,dataloader,device):

    model.train()

    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(dataloader.dataset)

def train_moe(model,optimizer,dataloader,device):

    model.train()

    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)+model.load_balance_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def test(model,dataloader,device):
    model.eval()

    total_correct = 0
    for data in dataloader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(dataloader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NCI1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--coef', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42,
                        help='seed number')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='seed number')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gnn_type',type=str,default='GIN_Moe',
                        choices=['GIN','GCN','GatedGCN','GIN_Moe','GCN_Moe','GatedGCN_Moe'])
    parser.add_argument("--dropout", type=float, default=.5,
                        help="input feature dropout")
    parser.add_argument("--fin_mlp_layers", type=int, default=1,
                        help="final mlp layers")
    parser.add_argument("--pooling", type=str, default="add")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--gate_type', type=str, default='GIN',
                        choices=['GIN','GCN','GAT','liner'])
    parser.add_argument('--min_layers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='results')

    args = parser.parse_args()
    logger = Logger(args.num_seeds,args.runs, args)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    dataset = get_dataset(args.dataset, normalize=args.normalize)
    train_loader=[]
    val_loader=[]
    test_loader=[]

    for i in range(1,11):
        train_idxes = torch.as_tensor(np.loadtxt('../datasets_split/%s/10fold_idx/train_idx-%d.txt' % (args.dataset, i),
                                                        dtype=np.int32), dtype=torch.long)
        val_idxes = torch.as_tensor(np.loadtxt('../datasets_split/%s/10fold_idx/test_idx-%d.txt' % (args.dataset, i),
                                                        dtype=np.int32), dtype=torch.long)
        test_idxes = torch.as_tensor(np.loadtxt('../datasets_split/%s/10fold_idx/test_idx-%d.txt' % (args.dataset, i),
                                                        dtype=np.int32), dtype=torch.long)
        
        train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))
        train_set, val_set, test_set = dataset[train_idxes], dataset[val_idxes], dataset[test_idxes]
        
        train_loader.append(DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True))
        val_loader.append(DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False))
        test_loader.append(DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False))

    seed_lst = [i+args.seed for i in range(args.num_seeds)]

    for seed in seed_lst:
        set_random_seed(seed)
        for run in range(args.runs):
            if args.gnn_type == 'GIN':
                model = GIN(
                    in_channels=dataset.num_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    fin_mlp_layers=args.fin_mlp_layers
                ).to(device)
            elif args.gnn_type == 'GCN':
                model = GCN(
                    in_channels=dataset.num_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    fin_mlp_layers=args.fin_mlp_layers
                ).to(device)
            elif args.gnn_type == 'GatedGCN':
                model = GatedGCN(
                    in_channels=dataset.num_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    fin_mlp_layers=args.fin_mlp_layers
                ).to(device)
            elif args.gnn_type == 'GIN_Moe':
                model = GNN_Moe(
                    in_channels=dataset.num_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    num_layers=args.num_layers,
                    topK=args.k,
                    model_type='GIN',
                    gate_type=args.gate_type,
                    min_layers=args.min_layers,
                    fin_mlp_layers=args.fin_mlp_layers,
                    coef = args.coef,
                    dropout=args.dropout,
                    device=device
                ).to(device)
            elif args.gnn_type == 'GCN_Moe':
                model = GNN_Moe(
                    in_channels=dataset.num_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    num_layers=args.num_layers,
                    topK=args.k,
                    model_type='GCN',
                    gate_type=args.gate_type,
                    min_layers=args.min_layers,
                    fin_mlp_layers=args.fin_mlp_layers,
                    coef = args.coef,
                    dropout=args.dropout,
                    device=device
                ).to(device)
            elif args.gnn_type == 'GatedGCN_Moe':
                model = GNN_Moe(
                    in_channels=dataset.num_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    num_layers=args.num_layers,
                    topK=args.k,
                    model_type='GatedGCN',
                    gate_type=args.gate_type,
                    min_layers=args.min_layers,
                    fin_mlp_layers=args.fin_mlp_layers,
                    coef = args.coef,
                    dropout=args.dropout,
                    device=device
                ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            for epoch in range(1, 1 + args.epochs):
                if 'Moe' in args.gnn_type:
                    loss = train_moe(model,optimizer,train_loader[run],device)
                else:
                    loss = train(model,optimizer,train_loader[run],device)
                train_acc = test(model,train_loader[run],device)
                val_acc = test(model,val_loader[run],device)
                test_acc = test(model,test_loader[run],device)
                
                scheduler.step()
                result = (train_acc, val_acc, test_acc)
                logger.add_result(seed-args.seed,run, result)

                if epoch % args.log_steps == 0:
                    train_acc, val_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:03d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.4f}%, '
                        f'Valid: {100 * val_acc:.4f}%, '
                        f'Test: {100 * test_acc:.4f}%')
            logger.print_statistics(args.dataset,'classification',args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.coef,args.k,seed-args.seed,run)
        logger.print_statistics(args.dataset,'classification',args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.coef,args.k,seed-args.seed)
    logger.print_statistics(args.dataset,'classification',args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.coef,args.k,)

if __name__ == "__main__":
    main()