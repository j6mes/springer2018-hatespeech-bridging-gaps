GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_1.py  > expt1.log
EMBEDDING=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_1.py  > expt1emb.log
EMBEDDING=1 LARGE=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_1.py  > expt1large.log

GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_2.py  > expt2.log
EMBEDDING=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_2.py  > expt2emb.log
EMBEDDING=1 LARGE=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_2.py  > expt2large.log

GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_3.py  > expt3.log
EMBEDDING=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_3.py  > expt3emb.log
EMBEDDING=1 LARGE=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_3.py  > expt3large.log

GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_5.py  > expt5.log
EMBEDDING=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_5.py  > expt5emb.log
EMBEDDING=1 LARGE=1 GPU=1 CUDA_DEVICE=2 PYTHONPATH=src TRAIN=1 python src/experiment/experiment_5.py  > expt5large.log
