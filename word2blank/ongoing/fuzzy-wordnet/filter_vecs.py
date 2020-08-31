import numpy as np
from tqdm import tqdm

vec_file = 'wiki-news-300d-1M.txt'
fileList = ['posList/nouns.txt','posList/verbs.txt','posList/adjectives.txt']
saveList = ['posVecs/wiki-news-300d-nouns.txt','posVecs/wiki-news-300d-verbs.txt','posVecs/wiki-news-300d-adjectives.txt']

vecDim = 300

def makeList(word_file):
	wordList = list()
	with open(word_file) as file:
		for line in file:
			wordList.append(line.rstrip('\n'))
	return wordList

nounList, verbList, adjList = [makeList(file) for file in fileList]
nounVecs, verbVecs, adjVecs = list(), list(), list() 
first = True
with open(vec_file) as file:
	for line in tqdm(file):
		if (not nounList) and (not verbList) and (not adjList) :
			break
		word = line.split(' ')[0].lower()
		if first:
			first = False
			continue
		elif word in nounList:
			nounVecs.append(line)
			nounList.remove(word)
		elif word in verbList:
			verbVecs.append(line)
			verbList.remove(word)
		elif word in adjList:
			adjVecs.append(line)
			adjList.remove(word)

nounVecs.insert(0,str(len(nounVecs))+' '+str(vecDim)+'\n')
verbVecs.insert(0,str(len(verbVecs))+' '+str(vecDim)+'\n')
adjVecs.insert(0,str(len(adjVecs))+' '+str(vecDim)+'\n')

posVecs = [nounVecs, verbVecs, adjVecs]
for i in range(len(posVecs)):
	with open(saveList[i],'w') as outfile:
		outfile.writelines(posVecs[i])






