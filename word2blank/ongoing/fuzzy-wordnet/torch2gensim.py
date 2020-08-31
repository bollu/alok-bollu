import torch 
import numpy as np
from tqdm import tqdm

model = torch.load('mammals.pth.best')
outfile = 'mammals.vec'

vecDim = 5
vecList = list(model['objects'])


for w in tqdm(model['objects']):
	vec = model['model']['lt.weight'][model['objects'].index(w)]
	line = list(np.asarray(model['model']['lt.weight'][model['objects'].index(w)]))
	line = [str(v) for v in line]
	line.insert(0,w)
	line = ' '.join(line)
	vecList.append(line+'\n')

vecList.insert(0,str(len(vecList))+' '+str(vecDim)+'\n')

first = True
with open(outfile,'w') as outfile:
	outfile.writelines(vecList)