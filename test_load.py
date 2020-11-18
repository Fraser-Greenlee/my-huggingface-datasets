import datasets

data = datasets.load_dataset('Fraser/news-category-dataset')

print(data['train'][0])
