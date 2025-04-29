#!/bin/bash
#SBATCH --job-name=jobNoSco   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=720:00:00   #720:00:00               # Time limit hrs:min:sec
#SBATCH --mem=120gb     # Job memory request
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56  # war 56 bei dem run 66318
#SBATCH --output=CPU_Frish_single%j.log   # Standard output and error log
#SBATCH --nodelist=gpunode09
#SBATCH --partition=GPU-A40


#GPU-A40      up   infinite      2  alloc gpunode[09-10]


echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/etc/profile.d/conda.sh" ]; then
        . "/etc/profile.d/conda.sh"
    else
        export PATH="/usr/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<



#echo "Running MLP training on the bench dataset of expitope from the hlaLigandAtlas and Protrans and focusing on f1 value of strong with sweep"


echo "Sweeping that sweet sweep sweatily"
echo ""
export LD_LIBRARY_PATH=/home/students/s.karg/.conda/envs/py39_gpu/lib/
export CUDA_VISIBLE_DEVICES=0

#source /home/students/s.karg/.bashrc
#CONDA_EXE=/usr/bin/conda

#$CONDA_EXE update conda -y
#$CONDA_EXE init

#source /home/students/s.karg/.bashrc
#conda init bash
conda activate py39_gpu

#python -m   pip install cudann

#wandb login --relogin
#wandb online


cd /proj/Expitope/BindingMLP-main

dir=splits_atlas_iedb/
dir=splits_atlas_protransonly #splits_iedb_protransonly splits_atlas_iedb_protransonly

#for data in atlas atlas_iedb iedb; do sbatch train_without_scores_frishman.sh $data ; done 


data=$1	
dir=splits_${data}_protransonly




#for data in atlas atlas_iedb atlas_iedb_final; do 
for data in atlas_iedb; do


dir=splits_${data}_under


echo "dir ist $dir"
echo "data is $data"

arch=gpu

#for i in {2..20} ; do
for i in {1..20} ; do
	cd /proj/Expitope/BindingMLP-main/data
	 mkdir -p splits_${data}_under/split_$i

	if [ ! -f splits_${data}_under/split_$i/val_data.tsv ]; then
		python splitter_${data}_under.py $i
	fi
        cd /proj/Expitope/BindingMLP-main

	python train_${data}_under_${arch}.py --val_path data/$dir/split_${i}/val_data.tsv --train_path data/$dir/split_${i}/train_data.tsv --test_path data/$dir/split_${i}/test_data.tsv  --num_workers $SLURM_CPUS_PER_TASK --batch_size 256 --learning_rate 0.0000539 --weight_decay 0.0001262 --num_layers 11 --dropout_prob 0.005113 --init_type kaiming  --num_nodes_per_layer 991

	#python train_fix_atlas_iedb_v2_sweepOnly.py --val_path data/$dir/split_${i}/val_data.tsv --train_path data/$dir/split_${i}/train_data.tsv --test_path data/$dir/split_${i}/test_data.tsv  --num_workers 1 --tune

	#python train_fix_atlas_iedb_v2_sweepOnly.py --val_path data/$dir/split_${i}/val_data.tsv --train_path data/$dir/split_${i}/train_data.tsv --test_path data/$dir/split_${i}/test_data.tsv  --num_workers 1 --tune
	#wandb agent bioinformatics_freising/uncategorized/kas7fhx0  # no protrans split 7 
       	#wandb agent bioinformatics_freising/uncategorized/nv7ca1bc  #506

	#wandb agent bioinformatics_freising/uncategorized/l0j7dwso  #354
	#wandb agent bioinformatics_freising/uncategorized/b32hucha
	#rm data/$dir/split_${i}/*
done
#done


done # for data

exit 1 


#CUDA_VISIBLE_DEVICES= "1" python -m torch.distributed.launch python train.py --val_path data/val_data.tsv --train_path data/train_data.tsv --test_path data/test_data.tsv  --num_workers 8 --batch_size 512

#for i in {4..20} ; do
#for i in {1..100} ; do
#wandb sweep sweep.yaml


#echo `sysctl kernel.shmall`
#echo `sysctl kernel.shmmax`

#exit 1 

export LD_LIBRARY_PATH=/home/students/s.karg/.conda/envs/py39_gpu/lib/
export CUDA_VISIBLE_DEVICES=0
#python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo ""
echo "BLIBLABLUB"
echo ""
python -c "import torch; print(torch.cuda.is_available())"

#exit 1 
#nvidia-smi

#python gputest.py
#exit 1

wandb agent sebastian-karg/sweep/wbl613gy
wand sweep --resume  sebastian-karg/sweep/wbl613gy
wandb agent sebastian-karg/sweep/wbl613gy
wand sweep --resume  sebastian-karg/sweep/wbl613gy
wandb agent sebastian-karg/sweep/wbl613gy
wand sweep --resume  sebastian-karg/sweep/wbl613gy
wandb agent sebastian-karg/sweep/wbl613gy
wand sweep --resume  sebastian-karg/sweep/wbl613gy
wandb agent sebastian-karg/sweep/wbl613gy
wand sweep --resume  sebastian-karg/sweep/wbl613gy

exit 1 

wandb agent sebastian-karg/sweep/q04qkxe3
wandb sweep --resume sebastian-karg/sweep/q04qkxe3
wandb agent sebastian-karg/sweep/q04qkxe3
wandb sweep --resume sebastian-karg/sweep/q04qkxe3
wandb agent sebastian-karg/sweep/q04qkxe3
wandb sweep --resume sebastian-karg/sweep/q04qkxe3
wandb agent sebastian-karg/sweep/q04qkxe3
wandb sweep --resume sebastian-karg/sweep/q04qkxe3
wandb agent sebastian-karg/sweep/q04qkxe3
wandb sweep --resume sebastian-karg/sweep/q04qkxe3

exit 1 

wandb agent sebastian-karg/sweep/m2cjp74y
wandb sweep --resume sebastian-karg/sweep/m2cjp74y
wandb agent sebastian-karg/sweep/m2cjp74y
wandb sweep --resume sebastian-karg/sweep/m2cjp74y
wandb agent sebastian-karg/sweep/m2cjp74y
wandb sweep --resume sebastian-karg/sweep/m2cjp74y
wandb agent sebastian-karg/sweep/m2cjp74y
#wandb agent sebastian-karg/sweep/bhl2z2o1
#wandb agent sebastian-karg/sweep/3k6vlsjc

#wandb agent sebastian-karg/sweep/pl7l9zem

exit 1



for i in 10 ; do #95 best one, before 10 1  
  #cd /home/students/s.karg/work/bindmlp/BindingMLP-main/data
  #python splitter_atlas_iedb_v2.py $i
  #cd /home/students/s.karg/work/bindmlp/BindingMLP-main

#    for lr in 1e-5  3e-7 ; do # hier einmal anders herum gestartet, weil abgebrochen 
#for lr in 3e-5 1e-5 3e-6 1e-6 3e-7 1e-7 ; do   
#      for wd in  1e-6 1e-4 0; do 
#         for nl in   2 4 ; do 
			#for nnpl in 128 512 ; do 
#           for dp in  0.005 0.05; do 
#	     for bs in 128 ; do
              # for nnpl in 512 ; do # tried 128 but failed somehow
#	         for it in normal xavier kaiming orthogonal; do 
                   #cd /home/students/s.karg/work/bindmlp/BindingMLP-main
		    python train_fix_atlas_iedb_v2_sweepOnly.py --val_path data/$dir/split_${i}/val_data.tsv --train_path data/$dir/split_${i}/train_data.tsv --test_path data/$dir/split_${i}/test_data.tsv  --num_workers 1 --tune
#                   python train_fix_atlas_iedb_v2_sweepOnly.py --val_path data/$dir/split_${i}/val_data.tsv --train_path data/$dir/split_${i}/train_data.tsv --test_path data/$dir/split_${i}/test_data.tsv  --num_workers 16 --batch_size $bs --learning_rate $lr --weight_decay $wd --num_layers $nl --dropout_prob $dp --init_type $it #--num_nodes_per_layer $nnpl
	           #rm data/$dir/split_${i}/*

	        #rm data/splits_atlas_protrans/split_${i}/*
#              done
#            done
#          done
#        done
#      done
#    done
#  done
done
#rm data/$dir/split_${i}/*
wandb sync

#python train.py --val_path data/val_data.tsv --train_path data/train_data.tsv --test_path data/test_data.tsv  --num_workers 8 --batch_size 512




