#!/usr/bin/python
#encoding: utf-8

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import sys
import pdb
import datetime
import csv
from operator import itemgetter
import argparse
import math
import re
import random
import scipy.stats
#import Gnuplot, Gnuplot.funcutils
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(add_help=False, description="""
    Calculate useful metrics about dataset.
    """)
parser.add_argument("-h", "--help", action="help",
    help=argparse.SUPPRESS)
parser.add_argument("--avg-file", required=True,
    type=argparse.FileType('r'),
    help="""Result of eventual filters and avg-ans-pt.""")

#####################################################################
 
def argmaxAndMax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])
    
#####################################################################    

def argminAndMin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])
    
#####################################################################

def avg(iterable):
    return sum(iterable)/len(iterable)
      
#####################################################################      
      
# PARSE AND ANALYSE DATASET FILE

args = parser.parse_args()

scores = {'compound':[], 'avgHeadMod':[], 'stdevHeadMod':[], 'avgHead':[],
          'stdevHead':[], 'avgMod':[], 'stdevMod':[], 'geommean':[], 
          'arithmean':[], 'nAnnot':[]}

STDEV_THRESH = 1.5
     
with args.avg_file as resultsfile :  
  reader = csv.DictReader(resultsfile, dialect=csv.excel_tab)
  for result in reader :
    scores['compound'].append(unicode(result["compound_surface"],encoding='utf8'))    
    scores['avgHeadMod'].append(float(result["compositionality"]))
    avgHead = float(result["avgHead"])
    avgMod  = float(result["avgModifier"])
    scores['avgHead'].append(avgHead)
    scores['avgMod'].append(avgMod)   
    scores['stdevHeadMod'].append(float(result["stdevHeadModifier"]))    
    scores['stdevHead'].append(float(result["stdevHead"]))
    scores['stdevMod'].append(float(result["stdevModifier"]))    
    scores['nAnnot'].append(float(result["nAnnot"]))       
    scores['geommean'].append(math.sqrt(avgHead*avgMod))
    scores['arithmean'].append((avgHead+avgMod)/2.0) 
   
n = len(scores["avgHeadMod"])
print("\nNumber of compounds in dataset: {}\n".format(n),file=sys.stderr)

print("Analysis of some values in the form avg / max(argmax) / min(argmin)",file=sys.stderr)
print("-------------------------------------------------------------------",file=sys.stderr)            

statcolumns = ['nAnnot','stdevHeadMod','stdevHead','stdevMod']
avgMaxMin = map(lambda y: (sum(y)/n,)+argmaxAndMax(y)+argminAndMin(y),
            map(lambda x: scores[x], statcolumns))
nHighStdev = map(lambda y: sum(map(lambda x: 1 if x > STDEV_THRESH else 0, y)),
             map(lambda x: scores[x] if x.startswith("stdev") else [], 
             statcolumns))

for (name,nhigh,avgmaxmin) in zip(statcolumns,nHighStdev,avgMaxMin) :
  print(("\nValues of {}: {:.2f} / {:.0f}({}) / {:.0f}({})".format(name,avgmaxmin[0],avgmaxmin[2],scores['compound'][avgmaxmin[1]],avgmaxmin[4],scores['compound'][avgmaxmin[3]])).encode("utf-8"),
        file=sys.stderr)
  if name.startswith("stdev") :
    print("Nb. of {} with stddev > {}: {} ({:.2f}%)".format(name,STDEV_THRESH,
          nhigh,nhigh*100.0/n),file=sys.stderr)

print("\nCorrelation and goodness of fit of compound vs. mean of components",file=sys.stderr)
print("------------------------------------------------------------------",file=sys.stderr)  

rho1 = scipy.stats.spearmanr(scores["avgHeadMod"],scores["geommean"])[0]
rho2 = scipy.stats.spearmanr(scores["avgHeadMod"],scores["arithmean"])[0]

linreg1 = scipy.stats.linregress(scores["avgHeadMod"],scores["geommean"])
linreg2 = scipy.stats.linregress(scores["avgHeadMod"],scores["arithmean"])

rdet1 = linreg1[2]**2
rdet2 = linreg2[2]**2

print("\nSpearman: compound vs. geom. mean: {}".format(rho1),file=sys.stderr)
print("Spearman: compound vs. arith. mean: {}".format(rho2),file=sys.stderr)
print("\nR2: compound vs. geom. mean: {}".format(rdet1),file=sys.stderr)
print("R2: compound vs. arith. mean: {}".format(rdet2),file=sys.stderr)

print("\nCorrelation between compound and head/modifier judgements",file=sys.stderr)
print("-----------------------------------------------------------",file=sys.stderr)  

rhohead = scipy.stats.spearmanr(scores["avgHeadMod"],scores["avgHead"])[0]
rhomod = scipy.stats.spearmanr(scores["avgHeadMod"],scores["avgMod"])[0]

rhead = scipy.stats.pearsonr(scores["avgHeadMod"],scores["avgHead"])[0]
rmod = scipy.stats.pearsonr(scores["avgHeadMod"],scores["avgMod"])[0]

print("\nSpearman: compound vs. head: {}".format(rhohead),file=sys.stderr)
print("Spearman: compound vs. modifier: {}".format(rhomod),file=sys.stderr)

print("\nPearson: compound vs. head: {}".format(rhead),file=sys.stderr)
print("Pearson: compound vs. modifier: {}".format(rmod),file=sys.stderr)

print("\nCorrelation between extremes and stddev (middle is harder?)",file=sys.stderr)
print("-----------------------------------------------------------",file=sys.stderr)  

rho3 = scipy.stats.spearmanr(scores["stdevHeadMod"],map(lambda x:-(x-2.5)**2+6,scores["avgHeadMod"]))[0]
print("\nSpearman: stddev compound vs. -(compound - 2.5)^2: {}".format(rho3),file=sys.stderr)

rdet3 = scipy.stats.linregress(scores["stdevHeadMod"],map(lambda x:-(x-2.5)**2,scores["avgHeadMod"]))[2]**2
print("R2: stddev compound vs. -(compound - 2.5)^2: {}".format(rdet3),file=sys.stderr)

plt.rc('font', family='serif')  
plt.plot(scores['avgHeadMod'],scores['arithmean'],'bo',label='$\otimes = $ arithmetic mean')
plt.plot(scores['avgHeadMod'],scores['geommean'],'ro',label='$\otimes = $ geometric mean')
plt.plot([0,5],[0*linreg2[0]+linreg2[1],5*linreg2[0]+linreg2[1]],'b-',label='Linear regression of geom. mean')
plt.plot([0,5],[0*linreg1[0]+linreg1[1],5*linreg1[0]+linreg1
[1]],'r-',label='Linear regression of arith. mean')
plt.xlabel('avg(comp($w_1, w_2$))')
plt.ylabel('avg(comp($w_1$)) $\otimes$ avg(comp($w_2$))')
plt.axis(xmin=0,xmax=5,ymin=0,ymax=5)
plt.legend(loc='lower right',fontsize='small')
plt.savefig(args.avg_file.name+"-cpd-avg-regression.pdf")

plt.cla()
plt.plot(scores["avgHeadMod"],scores["stdevHeadMod"],'bo',label='Avg. vs. stddev of compound score')
plt.plot(scores["avgHead"],scores["stdevHead"],'ro',label='Avg. vs. stddev of head score')
plt.plot(scores["avgMod"],scores["stdevMod"],'go',label='Avg. vs. stddev of modifier score')
plt.plot([0,5],[1.5,1.5],'k-',label='High stddev threshold')
plt.xlabel('avg(comp($w_1, w_2$))')
plt.ylabel('stddev(comp($w_1, w_2$))')
plt.legend(loc='best',fontsize='small')
plt.savefig(args.avg_file.name+"-avg-stdev-correl.pdf")

# Rankplot
plt.cla()
plt.rc('font', family='serif',size=16)  
sortscores = sorted(zip(scores["avgHeadMod"],scores["avgHead"],scores["avgMod"]))
sortscores = map(lambda x: (x[0],)+x[1],enumerate(sortscores))
#pdb.set_trace()
plt.plot(map(lambda x:x[0],sortscores),map(lambda x:x[1],sortscores),'bo',label='Compound')
plt.plot(map(lambda x:x[0],sortscores),map(lambda x:x[2],sortscores),'r^',label='Head')
plt.plot(map(lambda x:x[0],sortscores),map(lambda x:x[3],sortscores),'gs',label='Modifier')
plt.xlabel('Instances')
plt.ylabel('Average compositionality score')
plt.legend(loc='best',fontsize='large')
plt.savefig(args.avg_file.name+"-rankplot.pdf")


      

