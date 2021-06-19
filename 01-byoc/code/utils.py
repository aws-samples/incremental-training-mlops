import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

def plot_confusion_matrix(cm, target_names, folder_name,
                          title='Confusion matrix', cmap=None,
                          normalize=False):
    """Plot confustion matrix
        Save the confusion matrix as png file
    Args:
        cm ([[int]]): confusion matrix
        target_names ([str]): the label name of each class
        folder_name (str): filename of png file
        title (str)(optinal): the title name on png file
        cmap (cmap_type)(optional): the type of cmap
        normalize (bool)(optional): show the figure as percentage
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'
               .format(accuracy, misclass))
    plt.savefig(os.path.join(folder_name, "cfm.png"))
