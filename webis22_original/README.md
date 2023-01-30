# Webis Clickbait Spoiling Corpus

The Webis Clickbait Spoiling Corpus 2022 (Webis-Clickbait-22) contains 5,000 spoiled clickbait posts crawled from Facebook, Reddit, and Twitter.
This corpus supports the task of clickbait spoiling, which deals with generating a short text that satisfies the curiosity induced by a clickbait post.

This dataset contains the clickbait posts and manually cleaned versions of the linked documents, and extracted spoilers for each clickbait post.
Additionally, the spoilers are categorized into three types: short phrase spoilers, longer passage spoilers, and multiple non-consecutive pieces of text.

We want to organize a shared task on clickbait spoiling. Hence, we omit the 1,000 test post from this version of the dataset and will publish the test posts later.

## Overview

The dataset comes with predefined train/validation/test splits:

- [3,200 posts for training](training.jsonl)
- [800 posts for validation](validation.jsonl)
- [1,000 posts for testing](test.jsonl)
  - The test set is ommitted from this version of the dataset since we want to organize a shared task on clickbait spoiling and for this we want to keep the test set private until the end of the shared task.
- The [complete corpus with 5,000 clickbait posts](clickbait-spoiling-21.jsonl)
  - The clickbait-spoiling-21.jsonl file is ommitted from this version of the dataset since we want to organize a shared task on clickbait spoiling and for this we want to keep the test set private until the end of the shared task.
