import random

PREFIX = '../data/news-category/'

with open(PREFIX + 'all.json', 'r') as f:
    lines = f.read().split('\n')

all_data = []
for line in lines:
    all_data.append(line)

random.shuffle(all_data)

n_train_samples = round(len(all_data) * 0.8)
n_test_samples = round(len(all_data) * 0.15)

with open(PREFIX + 'train.json', 'w') as f:
    f.write(
        '\n'.join(all_data[:n_train_samples])
    )

with open(PREFIX + 'test.json', 'w') as f:
    f.write(
        '\n'.join(all_data[n_train_samples: n_train_samples + n_test_samples])
    )

with open(PREFIX + 'validation.json', 'w') as f:
    f.write(
        '\n'.join(all_data[n_train_samples + n_test_samples:])
    )
