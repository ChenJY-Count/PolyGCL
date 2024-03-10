python training.py --dataname cora --lr 0.0005 --wd 1e-3 --lr1 0.002 --dprate 0.3 --dropout 0.3
python training.py --dataname citeseer --lr 0.0001 --lr1 0.0005 --epochs 1000 --dprate 0.2 --dropout 0.3
python training.py --dataname pubmed --lr 0.0001 --lr1 0.001 --epochs 1000 --wd1 1e-3 --wd 1e-5 --is_bns True --act_fn prelu --dprate 0.6 --dropout 0.0
python training.py --dataname cornell --lr 0.0001 --dprate 0.8 --dropout 0.5
python training.py --dataname texas --lr 0.005 --dprate 0.4 --dropout 0.5
python training.py --dataname wisconsin --lr 0.0001 --is_bns True --act_fn prelu --dprate 0.1 --dropout 0.7
python training.py --dataname actor --lr 0.01 --is_bns True --act_fn prelu --dprate 0.3 --dropout 0.7
python training.py --dataname chameleon --lr 0.001 --patience 50 --epochs 1000 --is_bns True --act_fn prelu --dprate 0.3 --dropout 0.2
python training.py --dataname squirrel --lr 0.001 --is_bns True --act_fn prelu --dprate 0.2 --dropout 0.0