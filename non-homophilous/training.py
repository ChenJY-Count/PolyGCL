import argparse
import warnings
import seaborn as sns

import torch
from alive_progress import alive_bar
import random
import numpy as np
import torch as th
import torch.nn as nn

warnings.filterwarnings("ignore")

from model import LogReg,Model

parser = argparse.ArgumentParser(description="PolyGCL")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # Default seed same as GCNII
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--runs', type=int, default=5,help='number of distinct runs')

parser.add_argument('--dataset', type=str, default='fb100')
parser.add_argument('--sub_dataset', type=str, default='')
parser.add_argument(
    "--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu."
)
parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="Patient epochs to wait before early stopping.",
)
parser.add_argument(
    "--lr", type=float, default=0.010, help="Learning rate of prop."
)
parser.add_argument(
    "--lr1", type=float, default=0.001, help="Learning rate of PolyGCL."
)
parser.add_argument(
    "--lr2", type=float, default=0.01, help="Learning rate of linear evaluator."
)
parser.add_argument(
    "--wd", type=float, default=0.0, help="Weight decay of PolyGCL prop."
)
parser.add_argument(
    "--wd1", type=float, default=0.0, help="Weight decay of PolyGCL."
)
parser.add_argument(
    "--wd2", type=float, default=0.0, help="Weight decay of linear evaluator."
)

parser.add_argument(
    "--hid_dim", type=int, default=512, help="Hidden layer dim."
)

parser.add_argument(
    "--K", type=int, default=10, help="Layer of encoder."
)

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
th.manual_seed(args.seed)
th.cuda.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)

from dataset import load_nc_dataset
from data_utils import load_fixed_splits
from torch_geometric.utils import to_undirected

def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )

if __name__ == "__main__":
    print(args)
    ### Load and preprocess data ###
    dataset = load_nc_dataset(args.dataset, args.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(args.device)

    # load fixed dataset split
    split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)
    n_node = dataset.graph['num_nodes']
    # infer the number of classes for non one-hot and one-hot labels
    n_classes = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    n_feat = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(args.device), dataset.graph[
        'node_feat'].to(args.device)
    print(f"num nodes {n_node} | num classes {n_classes} | num node feats {n_feat}")

    feat = dataset.graph['node_feat']
    label = dataset.label.squeeze(1)
    edge_index = dataset.graph['edge_index']

    n_node = feat.shape[0]
    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))

    # Step 2: Create model =================================================================== #
    model = Model(n_feat, args.hid_dim)
    model = model.to(args.device)

    print(f"# params: {count_parameters(model)}")

    lbl = lbl.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam([{'params': model.encoder.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.disc.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                  {'params': model.encoder.prop1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
                                  {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr}
                                  ])

    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            shuf_idx = np.random.permutation(n_node)
            shuf_feat = feat[shuf_idx, :]

            out = model(edge_index, feat, shuf_feat)
            loss = loss_fn(out, lbl)

            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print("Epoch: {0}, Loss: {1:0.4f}".format(epoch, loss.item()))

            if loss < best:
                best = loss
                cnt_wait = 0
                th.save(model.state_dict(), "model_arxiv.pkl")
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping")
                break
            bar()

    model.load_state_dict(th.load("model_arxiv.pkl"))
    model.eval()
    embeds = model.get_embedding(edge_index, feat)

    print("=== Evaluation ===")
    ''' Linear Evaluation '''
    results = []

    for run in range(args.runs):
        assert label.shape[0] == n_node
        split_idx = split_idx_lst[run]
        train_idx, val_idx, test_idx = split_idx['train'].to(args.device), split_idx['valid'].to(args.device), split_idx['test'].to(args.device)

        train_embs = embeds[train_idx]
        val_embs = embeds[val_idx]
        test_embs = embeds[test_idx]

        train_labels = label[train_idx]
        val_labels = label[val_idx]
        test_labels = label[test_idx]
        assert len(torch.unique(train_labels)) == n_classes and len(torch.unique(val_labels)) == n_classes and len(torch.unique(test_labels)) == n_classes
        best_val_acc = 0
        eval_acc = 0
        bad_counter = 0

        logreg = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        logreg = logreg.to(args.device)

        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    bad_counter = 0
                    best_val_acc = val_acc
                    if test_acc > eval_acc:
                        eval_acc = test_acc
                else:
                    bad_counter += 1

        print(run, 'Linear evaluation accuracy:{:.4f}'.format(eval_acc))
        results.append(eval_acc.cpu().data)

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results, axis=0) * 100
    values = np.asarray(results, dtype=object)
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}')

