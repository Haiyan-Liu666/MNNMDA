
import numpy as np
from train import train
from sklearn.model_selection import KFold
from sklearn import metrics

if __name__ == "__main__":
  kf = KFold(5, shuffle=True)
  for train_index, test_index in kf.split(np.arange(450)):
    test_labels, scores = train(train_index, test_index)
    auroc = metrics.roc_auc_score(y_true=test_labels, y_score=scores)
    aupr = metrics.average_precision_score(y_true=test_labels, y_score=scores)
    print(f"auroc:{auroc}, aupr:{aupr}")
    print("*"*80)
