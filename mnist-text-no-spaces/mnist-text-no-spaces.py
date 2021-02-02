
"""MNIST text dataset with no spaces."""

from __future__ import absolute_import, division, print_function

import json
import os
import math

import numpy as np
import datasets


_DESCRIPTION = """\
MNIST dataset adapted to a text-based representation.

This allows testing interpolation quality for Transformer-VAEs.

System is heavily inspired by Matthew Rayfield's work https://youtu.be/Z9K3cwSL6uM

Works by quantising each MNIST pixel into one of 64 characters.
Every sample has an up & down version to encourage the model to learn rotation invarient features.

Use `.array_to_text(` and `.text_to_array(` methods to test your generated data.

Removed spaces to get better BPE compression on sequences.
**Should only be used with a trained tokenizer.**

Data format:
- text: (30 x 28 tokens, 840 tokens total): Textual representation of MNIST digit, for example:
```
00down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
01down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
02down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
03down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
04down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
05down!!!!!!!!!!!!!%%%@CL'Ja^@!!!!
06down!!!!!!!!(*8GK`````YL`]Q1!!!!
07down!!!!!!!-\\````````_855/*!!!!!
08down!!!!!!!%W`````RN^]!!!!!!!!!!
09down!!!!!!!!5H;``T#!+G!!!!!!!!!!
10down!!!!!!!!!$!G`7!!!!!!!!!!!!!!
11down!!!!!!!!!!!C`P!!!!!!!!!!!!!!
12down!!!!!!!!!!!#P`2!!!!!!!!!!!!!
13down!!!!!!!!!!!!)]YI<!!!!!!!!!!!
14down!!!!!!!!!!!!!5]``>'!!!!!!!!!
15down!!!!!!!!!!!!!!,O``F'!!!!!!!!
16down!!!!!!!!!!!!!!!%8``O!!!!!!!!
17down!!!!!!!!!!!!!!!!!_`_1!!!!!!!
18down!!!!!!!!!!!!!!,AN``T!!!!!!!!
19down!!!!!!!!!!!!*FZ```_N!!!!!!!!
20down!!!!!!!!!!'=X````S4!!!!!!!!!
21down!!!!!!!!&1V````R5!!!!!!!!!!!
22down!!!!!!%KW````Q5#!!!!!!!!!!!!
23down!!!!.LY````^B#!!!!!!!!!!!!!!
24down!!!!C```VBB%!!!!!!!!!!!!!!!!
25down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
26down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
27down!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```
- label: Just a number with the texts matching label.

"""

_CITATION = """\
@dataset{dataset,
    author = {Fraser Greenlee},
    year = {2021},
    month = {2},
    pages = {},
    title = {MNIST text dataset (no spaces).},
    doi = {}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/mnist-text-nospace/train.json.zip"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/mnist-text-nospace/test.json"

LABELS = list(range(10))


class MnistText(datasets.GeneratorBasedBuilder):
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
                pixels[y, x] = (ord(token) - 33) / 64

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
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(train_path, 'train.json')}
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
