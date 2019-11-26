import numpy as np
from sklearn import svm, metrics, model_selection
from svm_plot import plot_contours

spiral_X = np.load('spiral_X.npy')
spiral_Y = np.load('spiral_Y.npy')

X_train, X_test, y_train, y_test = model_selection.train_test_split(spiral_X, spiral_Y, test_size=0.2)
accuracy_scores = []
for i in range(1,10): #experimenting with different degrees of polynomial kernel
    clf = svm.SVC(kernel='poly', max_iter=100000, degree=i, gamma='auto')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(i, accuracy)
    accuracy_scores.append(accuracy)
    plot_contours(clf, spiral_X, spiral_Y)
print(accuracy_scores)

