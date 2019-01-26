import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets

good_features = np.load("good_features.npy")
bad_features = np.load("bad_features.npy")

X = np.concatenate([good_features[:, 0:3:2], bad_features[:,0:3:2]])  # we only take the first two features for visualization
Y = np.concatenate([np.ones(len(good_features)), np.zeros(len(bad_features))])

n_features = X.shape[1]

C = 10
kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

# Create different classifiers.
classifiers = {
    'L1 logistic': LogisticRegression(C=C, penalty='l1',
                                      solver='saga',
                                      multi_class='multinomial',
                                      max_iter=10000),
    'L2 logistic (Multinomial)': LogisticRegression(C=C, penalty='l2',
                                                    solver='saga',
                                                    multi_class='multinomial',
                                                    max_iter=10000),
    'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2',
                                            solver='saga',
                                            multi_class='ovr',
                                            max_iter=10000),
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                      random_state=0),
    'GPC': GaussianProcessClassifier(kernel)
}

n_classifiers = len(classifiers)

plt.figure(figsize=(3 * 2, n_classifiers * 2))
plt.subplots_adjust(bottom=.2, top=.95)

xx = np.linspace(0, 1, 100)
yy = np.linspace(0, 1, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X, Y)

    y_pred = classifier.predict(X)
    accuracy = accuracy_score(Y, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))

    # View probabilities:
    probas = classifier.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    # for k in range(n_classes):
    #     plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
    #     plt.title("Class %d" % k)
    #     if k == 0:
    #         plt.ylabel(name)
    #     imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
    #                                extent=(0, 1, 0, 1), origin='lower')
    #     # plt.xticks(())
    #     # plt.yticks(())
    #     idx = (y_pred == k)
    #     scatter_colors = []
    #     if idx.any():
    #         plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')

    plt.subplot(n_classifiers, n_classes, index+1)
    plt.ylabel(name)
    imshow_handle = plt.imshow(probas[:, 0].reshape((100, 100)),
                               extent=(0, 1, 0, 1), origin='lower')
    # plt.xticks(())
    # plt.yticks(())
    colors = np.where(Y == y_pred, 'g', 'r')
    markers = np.where(Y == 1, '^', 'v')
    labels = np.where(Y == 1, 'empty', 'not_empty')
    for x, y, c, m,l in zip(X[:, 0], X[:, 1], colors, markers,labels):
        plt.scatter(x, y, alpha=0.8, c=c, marker=m, edgecolor='k', label=l)

plt.legend(loc='upper left')
ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

plt.show()