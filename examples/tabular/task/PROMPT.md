Improve the accuracy of a classifier on the Adult (census income) tabular
dataset: predict whether income is ">50K" or "<=50K" from 14 numeric and
categorical columns, some with missing values.

train.py fits a scikit-learn model on the training split and prints the scores
returned by evaluate() from eval.py. The validation accuracy is the metric you
optimize. Constraints:
- Only use the training data from load_train_data(); never load the validation
  or test data or call functions from eval.py other than load_train_data and
  evaluate.
- train.py must print the scores returned by evaluate(model) unchanged. The
  model must accept the raw feature DataFrame in predict().
- Training and evaluation together must finish in under 60 seconds on a laptop
  CPU. Only scikit-learn, numpy and pandas are installed.
