"""News headlines and categories dataset."""

from __future__ import absolute_import, division, print_function

import datasets


_DESCRIPTION = """\
Copy of [Kaggle dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes), adding to Huggingface for ease of use.

Description from Kaggle:

Context

Generating humor is a complex task in the domain of machine learning, and it requires the models to understand the deep semantic meaning of a joke in order to generate new ones. Such problems, however, are difficult to solve due to a number of reasons, one of which is the lack of a database that gives an elaborate list of jokes. Thus, a large corpus of over 0.2 million jokes has been collected by scraping several websites containing funny and short jokes.

Visit my Github repository for more information regarding collection of data and the scripts used.

Content

This dataset is in the form of a csv file containing 231,657 jokes. Length of jokes ranges from 10 to 200 characters. Each line in the file contains a unique ID and joke.

Disclaimer

It has been attempted to keep the jokes as clean as possible. Since the data has been collected by scraping websites, it is possible that there may be a few jokes that are inappropriate or offensive to some people.
"""

_CITATION = None

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/short-jokes/train.json"


class ShortJokes(datasets.GeneratorBasedBuilder):
    """Short jokes dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'text': datasets.Value("string"),
                }
            ),
            homepage="https://github.com/Fraser-Greenlee/my-huggingface-datasets",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as txt_lines_file:
            data = []
            for line in txt_lines_file:
                data.append({'text': line})

            for id_, row in enumerate(data):
                yield id_, row
