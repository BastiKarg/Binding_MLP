# Binding_MLP
Description of the Training of the MLP models used for epitope binding classification in Expitope 3.0 

To start model training, first you need to unzip the data with: 

cd data
bzip2 -d atlas_iedb.tsv.bz2
cd ..


Training is started with modeltrain.sh. It is advised to pack this code in a slurm code to run it on your IT infrastructure. 
