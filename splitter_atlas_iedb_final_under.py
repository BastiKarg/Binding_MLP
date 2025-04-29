import csv
from sklearn.model_selection import train_test_split
import sys
import os
seed = int(sys.argv[1])
print("Seed is" + str(seed))

#exit(0)
#path = 'bench_data.tsv'
#path = "bench_atlas_iedb_v1.tsv"
#path = "bench_atlas_protrans.tsv_v3"
#path = "bench_atlas_iedb_protrans.tsv"
#path = "bench_atlas_protrans_fix1.tsv"
path = "atlas_iedb_final.tsv"

str_to_target = {'non': 0, 'weak': 1, 'strong': 2}

with open(path,'r') as f:
    reader = csv.reader(f, delimiter='\t')
    reader.__next__() # Skip header
    data = []
    for r in reader:
        #print(r)
        r2 = [float(r1) for r1 in r[:-1]]
        r2.append(str_to_target[r[-1]]) # Last col is target
        data.append(r2)
#%%
training_data, test_data = train_test_split(data, test_size=0.2, stratify=[d[-1] for d in data], random_state=seed)
train_data, val_data = train_test_split(training_data, test_size=0.2, stratify=[d[-1] for d in training_data], random_state=seed)

if not os.path.exists('splits_atlas_iedb_final_under/split_' + str(seed)):
    os.mkdir('splits_atlas_iedb_final_under/split_' + str(seed))

# Save splits
with open('splits_atlas_iedb_final_under/split_' + str(seed) + '/train_data.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(train_data)

with open('splits_atlas_iedb_final_under/split_' + str(seed) + '/val_data.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(val_data)

with open('splits_atlas_iedb_final_under/split_' + str(seed) + '/test_data.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(test_data)
