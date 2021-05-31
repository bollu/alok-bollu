import numpy as np
import os
import math
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

def get_emb(mat_file):
    f = open(mat_file, 'r')
    tmp = f.readlines()
    contents = tmp[1:]
    dimension = [int(x) for x in tmp[0].split(' ')]
    doc_emb = np.zeros((dimension[0], dimension[1]*dimension[1]))
    print(doc_emb.shape)
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        vec = tokens[1:]
        vec = np.array([float(ele) for ele in vec])
        mat = np.reshape(vec,(dimension[2], dimension[1]))
        sym_mat = mat.T@mat
        doc_emb[i] = np.reshape(sym_mat,-1)
    return doc_emb

def read_label(data_dir,label_file):
    f = open(os.path.join(data_dir, label_file))
    docs = f.readlines()
    y_true = np.array([int(doc.strip())-1 for doc in docs])
    labels = []
    for label in y_true:
        if label > 5:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    return labels

def f1(y_true, y_pred):
    assert y_pred.size == y_true.size
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    return f1_macro, f1_micro

def classifier(train, train_label, test, test_label):
    svclassifier = SGDClassifier()
    #svclassifier = MLPClassifier(hidden_layer_sizes=(100), activation='relu', max_iter=400)
    svclassifier.fit(train, train_label) 
    y_pred = svclassifier.predict(test)
    f1_macro, f1_micro = f1(test_label, y_pred)
    accuracy = accuracy_score(test_label, y_pred, normalize=True)
    print(f"F1 macro: {f1_macro}, F1 micro: {f1_micro}, Accuracy: {accuracy}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='svm-classify',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='aclImdb')
    parser.add_argument('--emb_file', default='imdb-emb1_3.txt')
    parser.add_argument('--train_num', default=25000, type=int)

    args = parser.parse_args()
    print(args)

    print(f'### Test: SVM Sentiment Classifier ###')
    doc_emb = get_emb(mat_file=os.path.join("datasets", args.dataset, args.emb_file))
    train_label = read_label(os.path.join("datasets", args.dataset),'train_lbl.txt')
    test_label = read_label(os.path.join("datasets", args.dataset),'test_lbl.txt')
    test = doc_emb[:args.train_num]
    rest = doc_emb[args.train_num:]
    train = rest[:args.train_num]
    classifier(train, train_label, test, test_label)
