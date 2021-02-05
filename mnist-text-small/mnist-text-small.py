
"""Compressed MNIST text dataset."""

from __future__ import absolute_import, division, print_function

import json
import os
import math

import numpy as np
import datasets


_DESCRIPTION = """\
MNIST dataset adapted to a text-based representation.

*Modified images to be ~1/4 the original area.*
Done by taking a max pool.

This allows testing interpolation quality for Transformer-VAEs.

System is heavily inspired by Matthew Rayfield's work https://youtu.be/Z9K3cwSL6uM

Works by quantising each MNIST pixel into one of 64 characters.
Every sample has an up & down version to encourage the model to learn rotation invarient features.

Use `.array_to_text(` and `.text_to_array(` methods to test your generated data.

Data format:
- text: (16 x 14 tokens, 224 tokens total): Textual representation of MNIST digit, for example:
```
00 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
01 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
02 down ! ! ! ! ! ! % % C L a ^ ! !
03 down ! ! ! - ` ` ` ` ` Y ` Q ! !
04 down ! ! ! % ` ` ` R ^ ! ! ! ! !
05 down ! ! ! ! $ G ` ! ! ! ! ! ! !
06 down ! ! ! ! ! # ` Y < ! ! ! ! !
07 down ! ! ! ! ! ! 5 ` ` F ! ! ! !
08 down ! ! ! ! ! ! ! % ` ` 1 ! ! !
09 down ! ! ! ! ! ! F ` ` ` ! ! ! !
10 down ! ! ! ! 1 ` ` ` ` 4 ! ! ! !
11 down ! ! L ` ` ` ` 5 ! ! ! ! ! !
12 down ! ! ` ` V B ! ! ! ! ! ! ! !
13 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
```
- label: Just a number with the texts matching label.

"""

_CITATION = """\
@dataset{dataset,
    author = {Fraser Greenlee},
    year = {2021},
    month = {1},
    pages = {},
    title = {MNIST small text dataset.},
    doi = {}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/mnist-text-small/train.json.zip"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/mnist-text-small/test.json"

LABELS = list(range(10))
CUSTOM_METHODS = ['array_to_text', 'text_to_array']
IMG_SIZE = (16, 14)


class MnistTextSmall(datasets.GeneratorBasedBuilder):
    """MNIST represented by text."""

    def as_dataset(self, *args, **kwargs):
        f"""
            Return a Dataset for the specified split.

            Modified to add custom methods {CUSTOM_METHODS} to the dataset.
            This allows rendering the text as images & vice versa.
        """
        a_dataset = super().as_dataset(*args, **kwargs)
        for method in CUSTOM_METHODS:
            setattr(a_dataset, f'custom_{method}', getattr(self, method))
        return a_dataset

    @staticmethod
    def array_to_text(pixels: np.array):
        '''
            Takes a 2D array of pixel brightnesses and converts them to text.
            Uses 64 tokens to represent all brightness values.
        '''
        width = pixels.shape[0]
        height = pixels.shape[1]

        lines = []

        for y in range(height):
            split = ['%02d down' % y]

            for x in range(width):
                brightness = pixels[y, x]

                mBrightness = math.floor(brightness * 64)
                s = chr(mBrightness + 33)

                split.append(s)

            lines.append(' '.join(split))

        reversed = []
        for line in lines:
            reversed.insert(0, (line.replace(' down ', ' up ', 1)))

        return ['\n'.join(lines), '\n'.join(reversed)]

    @staticmethod
    def text_to_array(text: str):
        '''
            Takes a text sequences and tries to convert it into a 2D numpy array of brightnesses.
            If parts of the text don't match the format they will be skipped.
        '''
        lines = text.split('\n')
        pixels = np.zeros((IMG_SIZE[1], IMG_SIZE[0] - 2))

        tokens = None
        for y, line in enumerate(lines):
            tokens = line.split(' ')
            for i in range(2, min(IMG_SIZE[0], len(tokens))):
                token = tokens[i]
                if len(token) == 1:
                    tkn_v = (ord(token) - 33)
                    if tkn_v >= 0 and tkn_v <= 64:
                        pixels[y, i - 2] = (ord(token) - 33) / 64

        if not lines:
            return pixels

        if tokens and len(tokens) > 1 and tokens[1] == 'up':
            pixels = pixels[::-1]

        return pixels

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'label': datasets.features.ClassLabel(names=LABELS),
                    'text': datasets.Value("string"),
                }
            ),
            homepage="https://github.com/Fraser-Greenlee/my-huggingface-datasets",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(train_path, 'train.json')}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as json_lines_file:
            data = []
            for line in json_lines_file:
                data.append(json.loads(line))

            for id_, row in enumerate(data):
                yield id_, row
