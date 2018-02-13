good config for InvertedPendulum-v1:
	python train_pg.py InvertedPendulum-v1 -n 100 -b 1000 -e 5 -rtg  --n_layers 2 --size 64 -lr 0.006 --exp_name lb_rtg_dna


HalfCaXX
	python others.py HalfCheetah-v1 -ep 150 --discount 0.9  -rtg -l 3 -s 32  -b 1000 -bl --exp_name wwg -lr 0.01
	python others.py HalfCheetah-v1 -ep 150 --discount 0.9  -rtg -l 3 -s 32  -b 3000 -bl --exp_name wwg -lr 0.01

	python train_pg.py HalfCheetah-v1 -ep 150 --discount 0.9  -rtg -l 4 -s 32  -b 3000 -bl --exp_name wwg2_my -lr 0.03

