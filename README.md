# kaggle-word2vec-movie-reviews
Kaggle-Bag of Words Meets Bags of Popcorn

This is the source code of my submission for the Kaggle competition "Bag of Words Meets Bags of Popcorn" (https://www.kaggle.com/c/word2vec-nlp-tutorial). The public leaderboard AUC score is 0.97568.

The model is **two-step ensemble model**. The first step is a weighted-avergae ensemble of *Bag-of-Words*, *Word2Vec*, *Doc2Vec* and *NBSVM* using logistic regression (denoted by WA). The second step is a weighted-average ensemble of WA and its two *modifications*.

Two modifications : 1) if the probability given by the average ensemble is greater than 0.5, the maximum probability of four differenct classifiers is chosen; if the probability given by the average ensemble is less than 0.5, the minimum probability of four differenct classifiers is chosen. 2) if the probability given by the weighted-average ensemble is greater than 0.5, the maximum probability of four differenct classifiers is chosen; if the probability given by the weighted-average ensemble is less than 0.5, the minimum probability of four differenct classifiers is chosen. The reason is that the output of the positive sample is as close to 1 as possible, and the output of the negative sample is as close to 0 as possible.

The performance of the two-step ensemble is **a little better** than that of the first ensemble.

# How to run #

* The code requires numpy, pandas, sklearn, bs4, nltk, and gensim.
* Firstly, generate word2vec and doc2vec models:
```
python generate_w2v.py
python generate_d2v.py
```
* After that, run the following command to generate the submission file (including the first step ensemble result):
```
python predict.py
```

