#!/usr/bin/env python3
import os
import json
import sys
import argparse
import pandas as pd
import os.path
from pandas import ExcelWriter

 

def parse(s):
    p = argparse.ArgumentParser()
    p.add_argument("fbaseline", help="baseline file")
    p.add_argument("fnew", help="new file")
    return p.parse_args(s)


def parsef(fp):
    """
    parse fp and return a dataframe
    file contains:

    capital-common-countries:
    ACCURACY TOP1: 1.58 %  (8 / 506)
    Total accuracy: 1.58 %   Semantic accuracy: 1.58 %   Syntactic accuracy: -nan % 
    repeated. 
    We want to extracted TOP1, total
    """
    ev = {}
    with open(fp, 'r') as f:
        ls = f.readlines()
    assert(ls is not None)
    for (i, l) in enumerate(ls[:-1]):
         if i % 3 == 0:
             n = l.strip()
             ev[n] = {}
         else:
             v = float(l.split(':')[1].split(' ')[1].strip())
             if i % 3 == 1:
                 ev[n]['top1'] = v
             else:
                 ev[n]['total'] = v
    return pd.DataFrame(ev)
if __name__ == "__main__":
    ps = parse(sys.argv[1:])
    base = parsef(ps.fbaseline)
    new = parsef(ps.fnew)
    df = pd.concat([base, new], keys=['base', 'new'])

    # keys to index axes with
    cols = df.axes[1]
    for c in cols:
        for t in ['top1', 'total']:
            delta =   df[c]['new'][t] - df[c]['base'][t] 
            if (delta > 0): 
                cell = df[c]['new'][t]
                cell = str(cell) + ' ↓'
            else: 
                cell = df[c]['new'][t]
                cell = str(cell) + ' ↑'
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(df)

