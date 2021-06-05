import os
import re

def preprocess(orig_file, typefile):
    print(orig_file)
    labels = []
    reviews = []
    for filename in os.listdir(orig_file+typefile+'/neg'):
        info = filename.split(".")[0]
        label = info.split("_")[1]
        labels.append(label)
        with open(orig_file+typefile+'/neg/'+filename,"r") as fin:
            content = fin.read()
            #lower case
            content = content.lower()
            #removing html tags
            content = re.sub('<.*?>',' ',content)
            #add space between punctuations and word, numbers
            content = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", content)
            content = re.sub(r"([\d/'+$\s]+|[^\d/'+$\s]+)\s*", r"\1 ", content)
            content = re.sub(r"(\w+)'s", r"\1 's", content)
            #substituting multiple spaces with single space
            content = re.sub('\s+',' ', content)
            reviews.append(content)
    for filename in os.listdir(orig_file+typefile+'/pos'):
        info = filename.split(".")[0]
        label = info.split("_")[1]
        labels.append(label)
        with open(orig_file+typefile+'/pos/'+filename,"r") as fin:
            content = fin.read()
            #lower case
            content = content.lower()
            #removing html tags
            content = re.sub('<.*?>',' ',content)
            #add space between punctuations and word, numbers
            content = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", content)
            content = re.sub(r"([\d/'+$\s]+|[^\d/'+$\s]+)\s*", r"\1 ", content)
            content = re.sub(r"(\w+)'s", r"\1 's", content)
            #substituting multiple spaces with single space
            content = re.sub('\s+',' ', content)
            reviews.append(content)
    if typefile == "train":
        for filename in os.listdir(orig_file+typefile+'/unsup'):
            with open(orig_file+typefile+'/unsup/'+filename,"r") as fin:
                content = fin.read()
                #lower case
                content = content.lower()
                #removing html tags
                content = re.sub('<.*?>',' ',content)
                #add space between punctuations and word, numbers
                content = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", content)
                content = re.sub(r"([\d/'+$\s]+|[^\d/'+$\s]+)\s*", r"\1 ", content)
                content = re.sub(r"(\w+)'s", r"\1 's", content)
                #substituting multiple spaces with single space
                content = re.sub('\s+',' ', content)
                reviews.append(content)
    with open(typefile+".txt","w+") as fout:
        for review in reviews:
            fout.write(review+"\n")
    with open(typefile+"_lbl.txt","w+") as fout:
        for label in labels:
            fout.write(label+"\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='sentiment-analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='dataset')
    #parser.add_argument('--emb_file', default='jose.txt')
    #parser.add_argument('--domain', default='headlines')

    args = parser.parse_args()
    preprocess(args.dataset, "train")
    preprocess(args.dataset, "test")
