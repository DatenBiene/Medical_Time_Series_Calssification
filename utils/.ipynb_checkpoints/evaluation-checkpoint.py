from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(ytrue,ypred):

    metrics = pd.DataFrame([],columns=['accuracy','recall','precision','f1-score'])
    metrics['accuracy'] = accuracy_score(ytrue,ypred)
    metrics['recall'] = recall_score(ytrue,ypred,average='macro')
    metrics['precision'] = precision_score(ytrue,ypred,average='macro')
    metrics['f1-score'] = f1_score(ytrue,ypred,average='macro')

    return metrics

def get_binary_metrics(ytrue,ypred):

    labels =  np.unique(ytrue)
    results = pd.DataFrame([],columns = ['recall','precision','f1-score'],
                            index = labels)
    for l in labels:
        ytrue_l = (ytrue==l).astype(int)
        ypred_l = (ypred == l).astype(int)
        results.loc[l] = [recall_score(ytrue_l,ypred_l),
                          precision_score(ytrue_l,ypred_l),
                          f1_score(ytrue_l,ypred_l)  ]
    return results
