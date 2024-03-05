# naive-bayes-classifier-example
> An example implementation of a Naïve Bayes classifier, for use in Machine Learning.

## Explaination
The Bayes' theorem can be represented by the following equation: 
$p(C|x) = \frac{p(x|C) \cdot p(C)}{p(x)}$

Where:
- \( p(x|C) \) is the probability of seeing the example/input \( x \) given the class \( C \).
- \( p(C|x) \) is the probability of the class \( C \) being correct given \( x \).
- \( p(x) \) is the prior probability of \( x \).
- \( p(w) \) is the prior probability of \( w \).

In Machine Learning, we typically wish to find the distribution of $ p(Ck|x) $ for every class k in order to find which class suits the example best.
Usually, finding $ p(Ck|x) $ is hard to calculate directly, so we use different approaches.
One approach, is to use Naïve Bayes for this classification, and calculating as-is the probabilities using the theorem.

Naïve Bayes is typically great for calculating the probability of sentences being in certain catagories.

In this repository, we have:
- r8-train-stemmed.txt and r8-test-stemmed.txt, which are datasets of sentences and the catagories they belong to.
- solution.py, that implements the classifier for the database, with high accuracy.

## Dependencies
the solution uses numpy, pandas and sklearn.
You can install them through:
     ```
     pip install -r requirements.txt
     ```
