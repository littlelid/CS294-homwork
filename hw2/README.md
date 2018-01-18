good config for InvertedPendulum-v1:
	python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -rtg  --n_layers 2 --size 64 -lr 0.006 --exp_name lb_rtg_dna
