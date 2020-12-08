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
Copy of [Kaggle dataset](https://www.kaggle.com/rmisra/news-category-dataset), adding to Huggingface for ease of use.

Description from Kaggle:

Context

This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. The model trained on this dataset could be used to identify tags for untracked news articles or to identify the type of language used in different news articles.

Content

Each news headline has a corresponding category. Categories and corresponding article counts are as follows:
```
POLITICS: 32739
WELLNESS: 17827
ENTERTAINMENT: 16058
TRAVEL: 9887
STYLE & BEAUTY: 9649
PARENTING: 8677
HEALTHY LIVING: 6694
QUEER VOICES: 6314
FOOD & DRINK: 6226
BUSINESS: 5937
COMEDY: 5175
SPORTS: 4884
BLACK VOICES: 4528
HOME & LIVING: 4195
PARENTS: 3955
THE WORLDPOST: 3664
WEDDINGS: 3651
WOMEN: 3490
IMPACT: 3459
DIVORCE: 3426
CRIME: 3405
MEDIA: 2815
WEIRD NEWS: 2670
GREEN: 2622
WORLDPOST: 2579
RELIGION: 2556
STYLE: 2254
SCIENCE: 2178
WORLD NEWS: 2177
TASTE: 2096
TECH: 2082
MONEY: 1707
ARTS: 1509
FIFTY: 1401
GOOD NEWS: 1398
ARTS & CULTURE: 1339
ENVIRONMENT: 1323
COLLEGE: 1144
LATINO VOICES: 1129
CULTURE & ARTS: 1030
EDUCATION: 1004
```

Acknowledgements

This dataset was collected from HuffPost.

Inspiration

Can you categorize news articles based on their headlines and short descriptions?
Do news articles from different categories have different writing styles?
A classifier trained on this dataset could be used on a free text to identify the type of language being used.
"""

_CITATION = """\
@dataset{dataset,
    author = {Misra, Rishabh},
    year = {2018},
    month = {06},
    pages = {},
    title = {News Category Dataset},
    doi = {10.13140/RG.2.2.20331.18729}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/news-category/train.json"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/news-category/test.json"
_VALIDATION_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/news-category/validation.json"

CATEGORIES = [
    'POLITICS',
    'WELLNESS',
    'ENTERTAINMENT',
    'TRAVEL',
    'STYLE & BEAUTY',
    'PARENTING',
    'HEALTHY LIVING',
    'QUEER VOICES',
    'FOOD & DRINK',
    'BUSINESS',
    'COMEDY',
    'SPORTS',
    'BLACK VOICES',
    'HOME & LIVING',
    'PARENTS',
    'THE WORLDPOST',
    'WEDDINGS',
    'WOMEN',
    'IMPACT',
    'DIVORCE',
    'CRIME',
    'MEDIA',
    'WEIRD NEWS',
    'GREEN',
    'WORLDPOST',
    'RELIGION',
    'STYLE',
    'SCIENCE',
    'WORLD NEWS',
    'TASTE',
    'TECH',
    'MONEY',
    'ARTS',
    'FIFTY',
    'GOOD NEWS',
    'ARTS & CULTURE',
    'ENVIRONMENT',
    'COLLEGE',
    'LATINO VOICES',
    'CULTURE & ARTS',
    'EDUCATION',
]


class NewsCategory(datasets.GeneratorBasedBuilder):
    """News headlines and categories dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'category_num': datasets.features.ClassLabel(names=CATEGORIES),
                    'category': datasets.Value("string"),
                    'headline': datasets.Value("string"),
                    'authors': datasets.Value("string"),
                    'link': datasets.Value("string"),
                    'short_description': datasets.Value("string"),
                    'date': datasets.Value("string"),
                }
            ),
            homepage="https://github.com/Fraser-Greenlee/my-huggingface-datasets",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        validation_path = dl_manager.download_and_extract(_VALIDATION_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as json_lines_file:
            data = []
            for line in json_lines_file:
                data.append(json.loads(line))

            for id_, row in enumerate(data):
                row['category_num'] = CATEGORIES.index(row['category'])
                yield id_, row
