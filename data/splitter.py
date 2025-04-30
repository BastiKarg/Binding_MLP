import csv
from sklearn.model_selection import train_test_split
import sys
import os
seed = int(sys.argv[1])
training_input = str(sys.argv[2])

print("Seed is" + str(seed))

path =   training_input + ".tsv"

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

training_data, test_data = train_test_split(data, test_size=0.2, stratify=[d[-1] for d in data], random_state=seed)
train_data, val_data = train_test_split(training_data, test_size=0.2, stratify=[d[-1] for d in training_data], random_state=seed)

if not os.path.exists(training_input):
    os.mkdir(training_input)

if not os.path.exists(training_input + '/split_' + str(seed)):
    os.mkdir(training_input + '/split_' + str(seed))

# Save splits
with open(training_input + '/split_' + str(seed) + '/train_data.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(train_data)

with open(training_input + '/split_' + str(seed) + '/val_data.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(val_data)

with open(training_input + '/split_' + str(seed) + '/test_data.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(test_data)
