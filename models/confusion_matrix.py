import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Compute and optionally normalize the confusion matrix.

    Parameters:
    - y_true: array-like of shape (n_samples,)
            True labels.
    - y_pred: array-like of shape (n_samples,)
            Predicted labels.
    - normalize: bool, optional (default=False)
            Whether to normalize the confusion matrix.

    Returns:
    - conf_matrix: ndarray of shape (n_classes, n_classes)
            Confusion matrix.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    if normalize:
        conf_matrix = conf_matrix.astype(
            'float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')
    print(conf_matrix)
    return conf_matrix


def plot_confusion_matrix(conf_matrix, classes, title, cmap, normalize=False):
    """
    This function prints and plots the confusion matrix.

    Parameters:
    conf_matrix (numpy.ndarray): The confusion matrix.
    classes (list): The list of class labels.
    title (str): The title of the plot.
    cmap (str or matplotlib.colors.Colormap): The colormap to use for the plot.
    normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.

    Returns:
    matplotlib.axes.Axes: The matplotlib axes object containing the plot.
    """
    fig, ax = plt.subplots()  # Add this line
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=0,
             ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)

    return ax


def draw_confusion_matrix(y_true, y_pred, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Greys):
    """
    Prints and draws the confusion matrix.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - classes (array-like): List of class labels.
    - normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
    - title (str, optional): Title of the confusion matrix plot. Default is "Confusion Matrix".
    - cmap (matplotlib colormap, optional): Colormap for the confusion matrix plot. Default is plt.cm.Greys.

    Returns:
    - ax (matplotlib Axes object): Axes object containing the confusion matrix plot.
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    if len(unique_labels) != len(classes):
        classes = unique_labels
    conf_matrix = compute_confusion_matrix(y_true, y_pred, normalize)
    ax = plot_confusion_matrix(conf_matrix, classes, title, cmap)
    return ax
