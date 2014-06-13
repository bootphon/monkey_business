import numpy as np
from sklearn import svm, linear_model
from extract_data import load_from_npz

fname = "blue_monkeys.npz"
fbanks, labels, strided, labels_strided, stride, labels_set = load_from_npz(
        fname)

select = np.logical_or(labels=="p", labels=="h") # TODO remove
fbanks = fbanks[select]
labels = labels[select]

clf = linear_model.LogisticRegression()
clf.fit(fbanks, labels)
print clf.score(fbanks, labels)

select_strided = np.logical_or(labels_strided=="p", labels_strided=="h") # TODO remove
strided = strided[select_strided]
labels_strided = labels_strided[select_strided]

#clf = svm.SVC()
clf = linear_model.LogisticRegression()
clf.fit(strided, labels_strided)
print clf.score(strided, labels_strided)

