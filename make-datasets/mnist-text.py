import math
from torchvision import datasets, transforms


def array_to_text(pixels):
    '''
        Takes a 2D array of pixel brightness, converts to text using 64 tokens to represent all brightness values.
    '''
    width = pixels.shape[0]
    height = pixels.shape[1]

    lines = []

    for y in range(height):
        split = ['%02d down' % y]

        for x in range(width):
            brightness = pixels[y, x]

            s = '~'

            mBrightness = math.floor(brightness * 64)
            s = chr(mBrightness + 33)

            split.append(s)

        lines.append(' '.join(split))

    reversed = []
    for line in lines:
        reversed.insert(0, (line.replace(' down ', ' up ', 1)))

    return [lines, reversed]


# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

for array, label in train_dataset:
    import pdb; pdb.set_trace()
    
