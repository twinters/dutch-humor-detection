# Dutch Humor Detection by Generating Negative Examples
![Python](https://img.shields.io/badge/python-v3.7-blue.svg?logo=Python&logoColor=white)

This repository includes the experiments we performed for our [paper »](). We have different features and multiple classifiers:

**Features**
- [TF-IDF »](src/datasets/tfidf.py)
- [Huggingface tokenizers »](src/datasets/tokenized.py)
- [Word embeddings »](src/datasets/word_embeddings.py)

**Classifiers**
- [LSTM »](src/modules/lstm.py)
- [Fully connected »](src/modules/linear.py)
- [CNN »](src/modules/cnn.py)
- [RobBERT (barebones) »](src/modules/robbert.py) and [RobBERT (all features, other repo)  »︎]()

### About
![Results](fig/results.png)

Detecting if a text is humorous is a hard task to do computationally, as it usually requires linguistic and common sense insights. In machine learning, humor detection is usually modelled as a binary classification task, trained to predict if the given text is a joke or another type of text. Rather than using completely different non-humorous texts, we propose using text generation algorithms for imitating the original joke dataset to increase the difficulty for the learning algorithm. We constructed several different joke and non-joke datasets to test the humor detection abilities of different language technologies. In particular, we test if the RobBERT language model is more capable than previous technologies for detecting humor when given generated similar non-jokes. In doing so, we create and compare the first Dutch humor detection systems. We found that RobBERT outperforms other algorithms, and especially shines when distinguishing jokes from the generated negative examples. This performance illustrates the usefulness of using text generation to create negative datasets for humor recognition, and also shows that transformer models are a large step forward in humor detection.


📄 Read the full paper [here »](https://arxiv.org/pdf/2010.13652.pdf)

### Get started
The code in `sources/` is used to scrape the data. See the corresponding files for each dataset.

There is also a script to prepare two scraped datasets for training a RobBERT model, you can use that script like this:

```shell script
python src/datasources/prepare_datsets.py data/processed/jokes.json data/processed/dynamic_template_jokes.json --mode classification 2> labels.txt 1> sentences.txt
```

The code in `src/` is to train the models. Run the training + evaluation code by running the entry point `run.py` in the root.

```shell script
pip install -r requirements.txt
python run.py
```

The code for Naive Bayes is on a separate branch, as it requires a different architecture than for Pytorch.

### Citing
```text
@inproceedings{winters2020humordetection,
  title={Dutch Humor Detection by Generating Negative Examples},
  author={Winters, Thomas and Delobelle, Pieter},
  booktitle={Proceedings of the 32st Benelux Conference on Artificial Intelligence (BNAIC 2020) and the 29th Belgian Dutch Conference on Machine Learning (Benelearn 2020)},
  year={2020},
  organization={CEUR-WS}
}
```
