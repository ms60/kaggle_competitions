import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


def check_feature(X,y,model,feature):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    baseline_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

    model = clone(model)
    