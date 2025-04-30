#!/bin/bash

#This script can be used to train models with an training data input of interest. Adjust parameters accordingly 

#set according to IT-infrastructure
#SLURM_CPUS_PER_TASK=

input=atlas_iedb # change if other data resource is used

# define range of seeds of interest
start_seed=1
end_seed=20

for seed in {${start_seed}..${end_seed}} ; do 
	cd data
	mkdir -p splits_${input}_under/split_$seed
	if [ ! -f splits_${input}_under/split_$seed/val_data.tsv ]; then
		python splitter.py $seed $input
	fi
	cd ..
	python train.py --val_path data/$input/split_${seed}/val_data.tsv --train_path data/$input/split_${seed}/train_data.tsv --test_path data/$input/split_${seed}/test_data.tsv
	# add the following parameters to achieve best performing model
	#--num_workers $SLURM_CPUS_PER_TASK --batch_size 256 --learning_rate 0.0000539 --weight_decay 0.0001262 --num_layers 11 --dropout_prob 0.005113 --init_type kaiming  --num_nodes_per_layer 991

done


