import argparse
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from model_ddi.gnn import GCN
from model_ddi.gnn_moe import GNN_Moe
from model_ddi.predictor import LinkPredictor
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import random
import numpy as np
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

def train(model, predictor, x, adj_t, split_edge, optimizer, batch_size):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss + model.load_balance_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbl-ddi')
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--coef', type=float, default=0.0005)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42,
                        help='seed number')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gnn_type',type=str,default='GCN_Moe',
                        choices=['GCN'])
    parser.add_argument("--dropout", type=float, default=.5,
                        help="input feature dropout")
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--gate_type', type=str, default='GCN',choices=['GCN','SAGE','liner'])
    parser.add_argument('--min_layers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    dataset = PygLinkPropPredDataset(name=args.dataset,
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}


    emb = torch.nn.Embedding(data.adj_t.size(0),
                             args.hidden_channels).to(device)

    evaluator = Evaluator(name=args.dataset)
    loggers = {
        'Hits@20': Logger(args.runs, args)
    }

    seed_lst = [i+args.seed for i in range(10)]
    for run in range(args.runs):
        set_random_seed(seed_lst[run])
        torch.nn.init.xavier_uniform_(emb.weight)

        if args.gnn_type == 'GCN':
            model = GCN(
                in_channels=args.hidden_channels, 
                hidden_channels=args.hidden_channels,
                out_channels= args.hidden_channels,
                num_layers=args.num_layers, 
                dropout=args.dropout).to(device)
        elif args.gnn_type == 'GCN_Moe':
            model = GNN_Moe(
                gnn_type='GCN',
                gate_type=args.gate_type,
                in_channels=args.hidden_channels, 
                hidden_channels=args.hidden_channels, 
                out_channels=args.hidden_channels,
                num_layers=args.num_layers,
                min_layers=args.min_layers,
                topK=args.k,
                coef=args.coef,
                dropout= args.dropout
            ).to(device)
            
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                args.num_layers, args.dropout).to(device)   
             
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, adj_t, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, adj_t, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
        loggers[key].print_statistics(args.dataset,args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.dropout,args.coef,args.k,args.seed,run)
    loggers[key].print_statistics(args.dataset,args.epochs,args.save_dir,args.gnn_type,args.gate_type,args.num_layers,args.min_layers,args.hidden_channels,args.dropout,args.coef,args.k,args.seed)


if __name__ == "__main__":
    main()