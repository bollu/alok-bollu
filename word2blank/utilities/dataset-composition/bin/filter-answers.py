#!/usr/bin/python
#encoding: utf-8

#from __future__ import unicode_literals
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
import numpy
import scipy.stats

numpy.seterr(all='ignore') # DANGEROUS!!!

parser = argparse.ArgumentParser(add_help=False, description="""
    Analyse CSV from our interface, average per compound.
    """)
parser.add_argument("-h", "--help", action="help",
    help=argparse.SUPPRESS)
parser.add_argument("--batch-file", required=True,
    type=argparse.FileType('r'),
    help="""Result downloaded from howisitgoing and CSVed with html2csv.sh.""")
parser.add_argument("--spearman-thresh", required=False,
    type=float,default=0.6,
    help="""Threshold below which we remove annotator (0.0 to 1.0, default 0.6)
Measure: spearman between annotator vs. avg. of other annotators""")
parser.add_argument("--zscore-thresh", required=False,
    type=float,default=3.0,
    help="""Threshold beyond which we remove annotation (0 to +inf, default 3.0)
Measure: Distance from avg. score in terms of nb. of stdev's""")
parser.add_argument("--lang", required=True,
    type=str,
    help="""Language of dataset (chose among 'pt', 'fr', 'en'""")

compoundavg = {}
userindex = {} 
compoundindex = {}
ANSTYPES = ['headmodifier','head','modifier']
for anstype in ANSTYPES : 
  compoundindex[anstype] = {} 
# Only French has a blacklist, but it's not a problem for other languages
blacklist = ['ATM829XNFM1CR', 'A3G6T12CCLQ5YU', 'A30QWUXRXRCE0Z',
             'A1XYKHLUQ6WZ9A', 'A1XI3KCNTXQ7J5', 'A1JWEYGDFDB100',
             'A2HQKQAXB8TAIV', 'A3MABNA5GE0VWE', 'AGVJXSVOYP8DQ', 
             'A3MABNA5GE0VWE', 'A29V34ENHFK8XP']

#####################################################################

def add_or_update(dictionary,key1,key2,value):
  entry1 = dictionary.get(key1,{})
  entry1[key2] = value
  dictionary[key1] = entry1

#####################################################################

def update_avg(compound, order, headlemma, modlemma, head,
               mod, ahead, amod, aheadmod, equivs):
  if ahead != "" and amod != "" and aheadmod != "":
    new = (None,None,"","",0,0,0,0,0,0,0,{})
    (cpdl,cpdlp,h,m,rh,rm,rhm,sh,sm,shm,n,equiv) = compoundavg.get(compound,new)
    if order == "AN":
      cpdl = modlemma + "_" + headlemma
      cpdlp = modlemma + "/a_" + headlemma + "/n"
    elif order == "NA":
      cpdl = headlemma + "_" + modlemma
      cpdlp = headlemma + "/n_" + modlemma + "/a"    
    n += 1
    rh += float(ahead)
    rm += float(amod)
    rhm += float(aheadmod)
    sh += float(ahead)**2
    sm += float(amod)**2
    shm += float(aheadmod)**2
    for eqi in equivs :
      if eqi.strip() != "{}" and eqi.strip() != "" and eqi.strip() != compound:
        equivcount = equiv.get(eqi,0)
        equivcount += 1
        equiv[eqi] = equivcount
    compoundavg[compound] = (cpdl,cpdlp,head,mod,rh,rm,rhm,sh,sm,shm,n,equiv)  

#####################################################################

def clean_equiv_pt(equiv):
  replacements = [("agua ","água "), ("voo ","vôo ")]
  equivres = re.sub(r"[ _\.-]+"," ",unicode(equiv.strip(),encoding="UTF-8").lower().encode(encoding="UTF-8"))
  equivres = re.sub(r"^(um|uma|o|a|os|as) ","",equivres)
  equivres = re.sub(r"(.)(ao)( |$)","\\1ão\\3",equivres)  
  equivres = re.sub(r"(^| )(refei|federa|situa|autoriza|op|repeti|aprova|constru)cão( |$)","\\1\\2ção\\3",equivres)    
  for (f, t) in replacements:
    equivres = equivres.replace(f, t)
  return equivres


#####################################################################

def clean_equiv_fr(equiv):
  replacements = [("Ã©","é"), ("Ã ","à"), ("Ã¢","â"), ("Ã´","ô"), ("Ãª","ê"),
                  ("Ã¨","è"), ("Ã®","î"), ("Ã§","ç"), ("Ã¯","ï"), ("Ã»","û"),
                  ("Ã¹","ù"), ("Ã¦","æ"), ("lâé","l'é"), ("dâé","d'é"),
                  ("dâi","d'i"), ("dâa","d'a"), ("dâo","d'o"),
                  ("paté ","pâté "), ("lâÃle", "l'Île"), ("Ãle ","Île "),
                  ("Ã","É"), ("dÄbit", "débit"), ("lâarrêt", "l'arrêt")]
  equivres = re.sub(r"[ _\.-]+"," ",equiv.strip().lower())
  equivres = re.sub(r"^(un|une|le|la|les) ","",equivres)
  equivres = re.sub(r"^l[']","",equivres)
  for (f, t) in replacements:
    equivres = equivres.replace(f, t)
  return equivres

#####################################################################

def clean_equiv_en(equiv):
  equivres = re.sub(r"[ _-]+"," ",equiv.strip().lower())
  equivres = re.sub(r"^(the|a|an) ","",equivres)
  return equivres

#####################################################################

def filter_outliers(userindex, compoundindex, zscore_thresh):
  global ANSTYPES
  removed_entries = total_entries = 0
  for user in userindex.keys() :
    for compound in userindex[user].keys() :
      total_entries += 1
      for anstype in ANSTYPES :
        userScore = float(userindex[user][compound]['Answer.Q'+anstype])
        #pdb.set_trace()
        allScores = compoundindex[anstype][compound].values()
        avgOthers = (sum(allScores)-userScore) / (len(allScores)-1)
        stdevOthers = math.sqrt((sum(map(lambda x:x*x,allScores))-userScore**2) \
                    / (len(allScores)-1) - avgOthers**2) + 0.000001
        zvalue = abs(userScore-avgOthers)/stdevOthers
        if zvalue > zscore_thresh :
          print("Removed outlier {}-{}, z-value: {:.2f}".format(compound,user,
                zvalue),file=sys.stderr)        
          del(userindex[user][compound])
          removed_entries += 1
          break
  print("Removed {}/{} ({:.2f}%) outliers\n\n".format(removed_entries, 
        total_entries, 100.0*removed_entries/total_entries),file=sys.stderr)

#####################################################################

def filter_annotators(userindex, compoundindex, spearman_thresh):
  global ANSTYPES
  removed_entries = total_entries = removed_annotators = 0
  total_annotators = len(userindex)  
  avg_rho = {}
  for at in ANSTYPES :
    avg_rho[at] = 0.0
  for user in userindex.keys() :
    uservec = userindex[user].keys()
    total_entries += len(uservec)
    if len(uservec)<=3:
      pass 
      # If not enough annotations, z-score filtering will have taken care of it
      # No point in applying this filter
      #print("User {} has only {} annotations, removing".format(user,
      #      len(uservec)),file=sys.stderr)
      #removed_entries += len(uservec)          
      #removed_annotators += 1
      #del(userindex[user])
    else :
      r = {}    
      for anstype in ANSTYPES : 
        userScores = map(lambda y:float(userindex[user][y]['Answer.Q'+anstype]),
                         uservec)
        avgothers = lambda (i,x): (sum(compoundindex[anstype][x].values()) \
                                  - userScores[i] ) \
                                  / (len(compoundindex[anstype][x])-1)
        avgScores = map(avgothers,enumerate(uservec))    


        r[anstype] = scipy.stats.spearmanr(userScores,avgScores)[0]
        if r[anstype] != float('nan') and r[anstype] < spearman_thresh :
          print("Removing all {} annotations of {}, rho={} on {}".format(
                len(uservec), user, r[anstype], anstype),file=sys.stderr)
          removed_entries += len(uservec)
          removed_annotators += 1        
          del(userindex[user])
          break
      if user in userindex.keys() :
        for at in r.keys() :
          if r[at] > -2 :
            avg_rho[at] += r[at]
  for at in r.keys() :
    avg_rho[at] /= (total_annotators-removed_annotators)
  
  print("Removed {}/{} ({:.2f}%) answers of {}/{} ({:.2f}%) annotators".format(
        removed_entries, total_entries, 100.0 * removed_entries / total_entries,
        removed_annotators, total_annotators, 
        100.0 * removed_annotators / total_annotators), file=sys.stderr)
  print("\nAverage rho with average of others:", file=sys.stderr)
  for at in avg_rho:
    print("  * {}: {:.2f}".format(at, avg_rho[at]), file=sys.stderr)
#####################################################################

def print_averages(compoundavg):
  totalequiv = 0
  print("\t".join(["compound_lemma","compound_surface","compound_lemmapos",
      "headLemma","modifierLemma","nAnnot","avgHead","stdevHead","avgModifier",
      "stdevModifier", "compositionality","stdevHeadModifier","equivalents"]))
  for compound in sorted(compoundavg.keys()) :  
    (cpdl,cpdlp,h,m,rh,rm,rhm,sh,sm,shm,n,equiv) = compoundavg[compound]
    # Averages
    ah = rh / n
    am = rm / n
    ahm = rhm / n
    # STD Devs
    dh = math.sqrt((sh - n*ah*ah) / (n-1))
    dm = math.sqrt((sm - n*am*am) / (n-1))
    dhm = math.sqrt((shm - n*ahm*ahm) / (n-1))
    cpds = compound.replace(" ","_")
    thresh_equiv = lambda x : "{}({})".format(x[0],x[1]) if x[1]>=1 else ""
    eqstr = re.sub(";+$","",";".join(map(thresh_equiv,sorted(equiv.items(),
                                                             reverse=True,
                                                             key=itemgetter(1)))))
    totalequiv += len(equiv.keys())
    template = "{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}"
    print (template.format(cpdl,cpds,cpdlp,h,m,n,ah,dh,am,dm,ahm,dhm,eqstr))

  print("Total equivalents: {}".format(totalequiv),file=sys.stderr)    

#####################################################################

def canonicalize(word):
  return word.replace("œ","oe").replace("-","_")

#####################################################################

args = parser.parse_args()
      
# PARSE AND ANALYSE BATCH FILE      
with args.batch_file as resultsfile :  
  if args.lang == 'pt': # PT comes from the HTML export
    reader = csv.DictReader(resultsfile, dialect=csv.excel_tab)
  else :
    reader = csv.DictReader(resultsfile)
  header = reader.fieldnames
  for result in reader :
    user = result["WorkerId"]  
    if result["AssignmentStatus"] == "Approved" and user not in blacklist :      
      compound = result["Input.compound"]
      add_or_update(userindex, user, compound, result)
      for anstype in ['headmodifier','head','modifier'] :      
        score = float(result["Answer.Q"+anstype])
        add_or_update(compoundindex[anstype], compound, user, score)  


number_initial = sum(map(lambda x: len(x),userindex.values()))
# REMOVE INDIVIDUAL ANNOTATIONS TOO FAR FROM AVERAGE
filter_outliers(userindex, compoundindex, args.zscore_thresh)        
# FILTER USERS TOO FAR FROM AVERAGE
filter_annotators(userindex, compoundindex, args.spearman_thresh)
# SOME STATS
number_final = sum(map(lambda x: len(x),userindex.values()))

print("Data retention rate {}/{}: {:.2%}".format(number_final,
      number_initial,float(number_final) / number_initial),file=sys.stderr)

# CALCULATE AVERAGES AND CLEAN EQUIVALENTS
for (user,annotations) in userindex.items() :
  for result in annotations.values() :   
    try : 
      head = result["Input.noun"]
    except KeyError:
      head = result["Input.head"]
    try :
      headlemma = result["Input.nounlemma"]
      modifierlemma = result["Input.modifierlemma"]
    except KeyError :
      headlemma = result["Input.head"]
      modifierlemma = result["Input.modifier"]      
    modifier = result["Input.modifier"]
    if args.lang == 'fr' :
      clean_equiv_lang = clean_equiv_fr
      equiv_list = map(lambda x:result["Answer.paraphrase"+str(x)],[1,2,3])
    elif args.lang == 'pt' :
      clean_equiv_lang = clean_equiv_pt
      equiv_list = result["equivalents"].split(" - ")    
    elif args.lang == 'en' :
      clean_equiv_lang = clean_equiv_en
      equiv_list = map(lambda x:result["Answer.paraphrase"+str(x)],[1,2,3])
    else :
      print("Language not supported : {}".format(args.lang))
      exit()
    update_avg(canonicalize(result["Input.compound"]), # problem oeil_rouge
               result.get("Input.order","AN"),
               canonicalize(headlemma),
               canonicalize(modifierlemma),
               head, modifier,
               result["Answer.Qhead"],
               result["Answer.Qmodifier"],
               result["Answer.Qheadmodifier"],
               map(clean_equiv_lang,equiv_list))
               
# PRINT OUTPUT               
print_averages(compoundavg)
