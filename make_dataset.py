import json
import random
import datasets


with open('News_Category_Dataset_v2.json', 'r') as f:
    lines = f.read().split('\n')

all_data = []
for line in lines:
    all_data.append(line)

random.shuffle(all_data)

n_train_samples = round(len(all_data) * 0.8)
n_test_samples = round(len(all_data) * 0.15)

with open('train.json', 'w') as f:
    f.write(
        '\n'.join(all_data[:n_train_samples])
    )

with open('test.json', 'w') as f:
    f.write(
        '\n'.join(all_data[n_train_samples: n_train_samples + n_test_samples])
    )

with open('validation.json', 'w') as f:
    f.write(
        '\n'.join(all_data[n_train_samples + n_test_samples:])
    )
