import os
import re

def preprocess(orig_file):
    print(orig_file)
    labels = []
    reviews = []
    for filename in os.listdir(orig_file+'/neg'):
        info = filename.split(".")[0]
        label = info.split("_")[1]
        labels.append(label)
        with open(orig_file+'/neg/'+filename,"r") as fin:
            content = fin.read()
            #lower case
            content = content.lower()
            #removing html tags
            content = re.sub('<.*?>',' ',content)
            #removing special characters (punctuation) '@,!' e.t.c.
            #content = re.sub('\W',' ', content) 
            #removing single characters
            content = re.sub('\s+[a-zA-Z]\s+',' ', content)
            #substituting multiple spaces with single space
            content = re.sub('\s+',' ', content)
            reviews.append(content)
    for filename in os.listdir(orig_file+'/pos'):
        info = filename.split(".")[0]
        label = info.split("_")[1]
        labels.append(label)
        with open(orig_file+'/pos/'+filename,"r") as fin:
            content = fin.read()
            #lower case
            content = content.lower()
            #removing html tags
            content = re.sub('<.*?>',' ',content)
            #removing special characters (punctuation) '@,!' e.t.c.
            #content = re.sub('\W',' ', content) 
            #removing single characters
            content = re.sub('\s+[a-zA-Z]\s+',' ', content)
            #substituting multiple spaces with single space
            content = re.sub('\s+',' ', content)
            reviews.append(content)
    # for filename in os.listdir(orig_file+'/unsup'):
    #     with open(orig_file+'/unsup/'+filename,"r") as fin:
    #         content = fin.read()
    #         #lower case
    #         content = content.lower()
    #         #removing html tags
    #         content = re.sub('<.*?>',' ',content)
    #         #removing special characters (punctuation) '@,!' e.t.c.
    #         #content = re.sub('\W',' ', content) 
    #         #removing single characters
    #         content = re.sub('\s+[a-zA-Z]\s+',' ', content)
    #         #substituting multiple spaces with single space
    #         content = re.sub('\s+',' ', content)
    #         reviews.append(content)
    with open("test.txt","w+") as fout:
        for review in reviews:
            fout.write(review+"\n")
    with open("test_lbl.txt","w+") as fout:
        for label in labels:
            fout.write(label+"\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='sentiment-analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='aclImdb')
    #parser.add_argument('--emb_file', default='jose.txt')
    #sparser.add_argument('--domain', default='headlines')

    args = parser.parse_args()
    preprocess(orig_file=os.path.join("./datasets", args.dataset))