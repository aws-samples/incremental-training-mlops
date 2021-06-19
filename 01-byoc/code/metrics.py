from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score

def roc_auc(y_true, y_score, multi_class='ovr', average='weighted'):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    Args:
        y_true: array-like of shape (n_samples,) or (n_samples, n_classes)
        y_score: array-like of shape (n_samples,) or (n_samples, n_classes)
        average: {‘micro’, ‘macro’, ‘samples’, ‘weighted’} or None, default=’macro’
                        'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.
                        'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                        'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
        multi_class: {‘raise’, ‘ovr’, ‘ovo’}, default=’raise’
                        Stands for One-vs-rest. Computes the AUC of each class against the rest
                        Stands for One-vs-one. Computes the average AUC of all possible pairwise combinations of classes
    """
    return roc_auc_score(y_true, y_score, multi_class=multi_class, average=average)


def accuracy(y_true, y_pred, normalize=True, sample_weight=None):
    """ Accuracy classification score.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        normalize: bool, optional (default=True)
        sample_weight: array-like of shape (n_samples,), default=None
    Return:
        accuracy score: float
    """
    return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)


def f1(y_true, y_pred, pos_label=0, average='macro'):
    """ F1 = 2 * (precision * recall) / (precision + recall).
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        pos_label: str or int, 1 by default. setting labels=[pos_label] and average != 'binary' will report scores for that label only.
        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
    Return:
        f1 score: float
    """
    return f1_score(y_true, y_pred, average=average)


def precision(y_true, y_pred, pos_label=0, average='macro'):
    """ the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        pos_label: str or int, 1 by default. setting labels=[pos_label] and average != 'binary' will report scores for that label only.
        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
    Return:
        precision score: float
    """
    return precision_score(y_true, y_pred, average=average)


def recall(y_true, y_pred, pos_label=0, average='macro'):
    """ the ratio tp / (tp + fn) where tp is the number of true positives and fp the number of false negatives.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        pos_label: str or int, 1 by default. setting labels=[pos_label] and average != 'binary' will report scores for that label only.
        average: string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
    Return:
        recall score: float
    """
    return recall_score(y_true, y_pred, average=average)


def cfm(y_true, y_pred):
    """ Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
    Return:
        confusion matrix: ndarray of shape (n_classes, n_classes)
    """
    return confusion_matrix(y_true, y_pred)


def classification_report(y_true, y_pred, target_names=["Barking", "Howling", "Crying", "COSmoke","GlassBreaking","Other"]):
    """ Build a text report showing the main classification metrics.
    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix
        y_pred: 1d array-like, or label indicator array / sparse matrix
        target_names: list of strings. display names matching the labels (same order).
    Return:
        report: string
    """
    return metrics.classification_report(y_true, y_pred, target_names=target_names, digits=3)
