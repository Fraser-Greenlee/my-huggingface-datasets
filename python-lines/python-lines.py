# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""News headlines and categories dataset."""

from __future__ import absolute_import, division, print_function

import json

import datasets


_DESCRIPTION = """\
Dataset of single lines of Python code taken from the [CodeSearchNet](https://github.com/github/CodeSearchNet) dataset.

Context

This dataset allows checking the validity of Variational-Autoencoder latent spaces by testing what percentage of random/intermediate latent points can be greedily decoded into valid Python code.

Content

Each row has a parsable line of source code.
{'text': '{python source code line}'}

Most lines are < 100 characters while all are under 125 characters.

Contains 2.6 million lines.

All code is in parsable into a python3 ast.

"""

_CITATION = """\
@dataset{dataset,
    author = {Fraser Greenlee},
    year = {2020},
    month = {12},
    pages = {},
    title = {Python single line dataset.},
    doi = {}
}
"""


class PythonLines(datasets.GeneratorBasedBuilder):
    """Python lines dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'text': datasets.Value("string"),
                }
            ),
            homepage="",
            citation=_CITATION,
        )

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as json_lines_file:
            data = []
            for line in json_lines_file:
                data.append(json.loads(line))

            for id_, row in enumerate(data):
                yield id_, row
