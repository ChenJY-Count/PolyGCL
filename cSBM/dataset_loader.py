import torch
import pickle
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
import os
from cSBM_dataset import dataset_ContextualSBM

class Chame_Squir_Actor(InMemoryDataset):
    def __init__(self, root='data/', name=None, p2raw=None, transform=None, pre_transform=None):
        if name =='actor':
            name = 'film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(Chame_Squir_Actor, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data')
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

def DataLoader(name):
    if 'cSBM' in name:
        path = './data/'
        dataset = dataset_ContextualSBM(root=path, name=name)
    else:
        name = name.lower()

        if name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(osp.join('./data', name), name, transform=T.NormalizeFeatures())
        elif name in ['chameleon', 'actor', 'squirrel']:
            dataset = Chame_Squir_Actor(root='./data/', name=name, transform=T.NormalizeFeatures())
        elif name in ['texas', 'cornell', 'wisconsin']:
            dataset = WebKB(root='./data/',name=name, transform=T.NormalizeFeatures())
        else:
            raise ValueError(f'dataset {name} not supported in dataloader')
    return dataset

if __name__ == "__main__":
    name = 'chameleon'
    dataset = DataLoader(name=name)

    # print(dataset)
    # print(dir(dataset))
    # print(dataset.data)
    print(dataset.num_classes)
    print(dataset.num_features)
    data = dataset[0]
    print(data)
    print(data.train_percent)
    print(data.x)
    print('-'*30)


    # print(data1.train_percent)

   # For directed setting
   # dataset = WikipediaNetwork(root=root, name=name, transform=T.NormalizeFeatures())
   # For GPRGNN-like dataset, use everything from "geom_gcn_preprocess=False" and
   # only the node label y from "geom_gcn_preprocess=True"
    root = './data1/'
    preProcDs = WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
    dataset1 = WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    data1 = dataset1[0]
    data1.edge_index = preProcDs[0].edge_index
    dataset1.data.edge_index = preProcDs[0].edge_index
    print(dataset1.num_classes)
    print(dataset1.num_features)
    print(data1)
    print('ok')
    # print(is_undirected(edge_index=))
    print(data1.x)