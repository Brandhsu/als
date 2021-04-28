import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve

# TODO: sensitivity & specificity of model similar scores for both

def class_one_acc(labels):
    return sum(labels)/len(labels)

def acc(true, preds):
    return accuracy_score(true, preds)

def roc_auc(true, conf_scores, verbose=False):
    fpr, tpr, _ = roc_curve(true, conf_scores)
    return auc(fpr, tpr) if not verbose else (fpr, tpr, auc(fpr, tpr))

def prc_auc(true, conf_scores, verbose=False):
    precision, recall, _ = precision_recall_curve(true, conf_scores)
    return auc(recall, precision) if not verbose else (recall, precision, auc(recall, precision))

def plot_auc(true, conf_scores, mode='roc', lw=2):
    if mode == 'roc':   
        metric = roc_auc
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate'
        title = 'ROC AUC'
        p1 = [0, 1]
        p2 = [0, 1]
    elif mode == 'prc': 
        metric = prc_auc
        xlabel = 'Recall'
        ylabel = 'Precision'
        title = 'PRC AUC'
        p1 = [0, 1]
        p2 = [class_one_acc(true), class_one_acc(true)]
    else: return;
    
    scores = metric(true, conf_scores, verbose=True)

    plt.figure()
    plt.plot(scores[0], scores[1], color='red', lw=lw, label='ROC curve (area = %0.4f)' % scores[2])
    plt.plot(p1, p2, color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return scores[2]