import argparse
import torch
import sys
sys.path.append('../')
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model.gnn import GNN_OGB
from model.gnn_moe import GNN_Moe_OGB
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from logger import Logger
import numpy as np
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model,optimizer,dataloader,task_type,device,use_edge_attr=False):

    model.train()

    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        if use_edge_attr:
            out = model(data)
        else:
            out = model(data)
        is_labeled = data.y == data.y
        if "classification" in task_type: 
            loss = F.binary_cross_entropy_with_logits(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
        else:
            loss = F.mse_loss(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(dataloader.dataset)

def train_moe(model,optimizer,dataloader,task_type,device,use_edge_attr=False):

    model.train()

    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        if use_edge_attr:
            out = model(data)
        else:
            out = model(data)
        is_labeled = data.y == data.y
        if "classification" in task_type: 
            loss = F.binary_cross_entropy_with_logits(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled]) + model.load_balance_loss
        else:
            loss = F.mse_loss(out.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled]) + model.load_balance_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(dataloader.dataset)



@torch.no_grad()
def test(model,dataloader,evaluator,device,use_edge_attr=False):
    model.eval()

    # total_correct = 0
    y_true=torch.tensor([], device=device)
    y_pred=torch.tensor([], device=device)
    for data in dataloader:
        data = data.to(device)
        if use_edge_attr:
            out = model(data)
        else:
            out = model(data)
        # pred = out.argmax(dim=-1)
        y_pred=torch.cat([y_pred,out],0)
        y_true=torch.cat([y_true,data.y.view(out.shape)],0)
        # total_correct += int((pred == data.y).sum())
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict) # E.g., {"rocauc": 0.7321} 
    return result_dict


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ogbg-molbbbp')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42,
                        help='seed number')
    parser.add_argument('--coef', type=float, default=0.001)
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='seed number')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--gnn_type',type=str,default='GIN',
                        choices=['GIN','GCN','GIN_Moe','GCN_Moe'])
    parser.add_argument("--dropout", type=float, default=.5,
                        help="input feature dropout")
    parser.add_argument("--fin_mlp_layers", type=int, default=1,
                        help="final mlp layers")
    parser.add_argument("--pooling", type=str, default="maen")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--gate_type', type=str, default='GIN',
                        choices=['GIN','GCN','GAT','liner'])
    parser.add_argument('--min_layers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

    args = parser.parse_args()
    logger = Logger(args.num_seeds,1, args)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    dataset = PygGraphPropPredDataset(name = args.dataset, root = '../dataset/')

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,num_workers = args.num_workers,drop_last=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,num_workers = args.num_workers)


    seed_lst = [i+args.seed for i in range(args.num_seeds)]
    use_edge_attr =False
    for seed in seed_lst:
        set_random_seed(seed)
        if args.gnn_type == 'GIN':
            model = GNN_OGB(
                num_tasks=dataset.num_tasks,
                num_layer=args.num_layers,
                emb_dim=args.emb_dim,
                gnn_type='gin',
                drop_ratio=args.dropout
                ).to(device)
            use_edge_attr = True
        if args.gnn_type == 'GCN':
            model = GNN_OGB(
                num_tasks=dataset.num_tasks,
                num_layer=args.num_layers,
                emb_dim=args.emb_dim,
                gnn_type='gcn',
                drop_ratio=args.dropout
                ).to(device)
            use_edge_attr = True
        elif args.gnn_type == 'GIN_Moe':
            model = GNN_Moe_OGB(
                device=device,
                emb_dim=args.emb_dim,                    
                num_tasks=dataset.num_tasks,
                num_layer=args.num_layers,
                topK=args.k,
                gnn_type='gin',
                gate_type=args.gate_type,
                min_layers=args.min_layers,
                coef=args.coef,
                drop_ratio=args.dropout
            ).to(device)
            use_edge_attr = True
        elif args.gnn_type == 'GCN_Moe':
            model = GNN_Moe_OGB(
                device=device,
                emb_dim=args.emb_dim,                    
                num_tasks=dataset.num_tasks,
                num_layer=args.num_layers,
                topK=args.k,
                gnn_type='gcn',
                gate_type=args.gate_type,
                min_layers=args.min_layers,
                coef=args.coef,
                drop_ratio=args.dropout
            ).to(device)

        evaluator = Evaluator(name = args.dataset)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        for epoch in range(1, 1 + args.epochs):
            if 'Moe' in args.gnn_type:
                loss = train_moe(model,optimizer,train_loader,dataset.task_type,device,use_edge_attr)
            else:
                loss = train(model,optimizer,train_loader,dataset.task_type,device,use_edge_attr)


            train_curve = test(model,train_loader,evaluator,device,use_edge_attr)[dataset.eval_metric]
            val_curve = test(model,valid_loader,evaluator,device,use_edge_attr)[dataset.eval_metric]

            test_curve = test(model,test_loader,evaluator,device,use_edge_attr)[dataset.eval_metric]
            scheduler.step()
            result = (train_curve, val_curve, test_curve)
            # print(result)
            logger.add_result(seed-args.seed,0, result)

            if epoch % args.log_steps == 0:
                train_curve, val_curve, test_curve = result
                if 'classification' in dataset.task_type:
                    print(f'Seed: {seed:02d}, '
                        f'Epoch: {epoch:03d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_curve:.4f}%, '
                        f'Valid: {100 * val_curve:.4f}%, '
                        f'Test: {100 * test_curve:.4f}%')
                else:
                    print(f'Seed: {seed:02d}, '
                        f'Epoch: {epoch:03d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {train_curve:.4f}, '
                        f'Valid: {val_curve:.4f}, '
                        f'Test: {test_curve:.4f}')
        logger.print_statistics(args.dataset,dataset.task_type,args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.emb_dim,args.coef,args.k,seed-args.seed)
    logger.print_statistics(args.dataset,dataset.task_type,args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.emb_dim,args.coef,args.k)

if __name__ == "__main__":
    main()