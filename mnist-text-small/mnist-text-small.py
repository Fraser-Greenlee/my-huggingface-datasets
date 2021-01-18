
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

Works by quantising each MNIST pixel into on of 64 characters.
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


class MnistTextSmall(datasets.GeneratorBasedBuilder):
    """MNIST represented by text."""
    def array_to_text(pixels: np.array):
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

        return ['\n'.join(lines), '\n'.join(reversed)]

    def text_to_array(text: str):
        lines = text.split('\n')
        pixels = np.zeros((len(lines), len(lines[0].split(' ')) - 2))

        for y, line in enumerate(lines):
            tokens = line.split(' ')
            assert(tokens[1] == 'down')
            pixel_tokens = tokens[2:]
            for x, token in enumerate(pixel_tokens):
                pixels[y, x] = ord(token) - 33

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
