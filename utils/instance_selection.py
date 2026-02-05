import os 
from pathlib import Path
import random 




path = Path.home() / '2omf' / 'instances' / 'new_batch'
files = [f for f in os.listdir(path) if f.endswith('.dat')]
files.sort()

print(f'There are {len(files)} instances in the folder {path}')

# get the training test and test set. 
instance_types = ['uniform', 'monarchy', 'diarchy', 'limulus']
training_set = []
test_set = []

# draw 60% of the instances for the training set
training_set = random.sample(files, k=int(0.6*len(files)))
training_set.sort()
# draw the rest for the test set
test_set = [f for f in files if f not in training_set]
test_set.sort()

print(f'There are {len(training_set)} instances in the training set')
print(f'There are {len(test_set)} instances in the test set')



# print it to file 'instances-list.txt'
with open('instances-list.txt', 'w') as f:
    for instance in training_set:
        f.write(f'{instance}\n')

with open('test-instances-list.txt', 'w') as f:
    for instance in test_set:
        f.write(f'~/2omf/files/instances/new_batch/{instance}\n')