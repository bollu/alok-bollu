Compositionality of Nominal Compounds - Datasets
------------------------------------------------

_Silvio Cordeiro, Carlos Ramisch, Aline Villavicencio, Leonardo Zilio, Marco Idiart, Rodrigo Wilkens_
_Version 1.0 - August 2, 2016_

This package contains numerical judgements by human native speakers about 180 nominal compound compositionality in English (EN), French (FR) and Brazilian Portuguese (PT).

Judgements were obtained using Amazon Mechanical Turk (EN and FR) and a web interface for volunteers (PT). Every compound has 3 scores: compositionality of head word, compositionality of modifier word and compositionality of the whole. Scores range from 1 (fully idiomatic) to 5 (fully compositonal) and are averaged over several annotators (around 10 to 20 depending on the language). All compounds in FR and PT, and 90 compounds in EN, also have synonyms and similar expressions given by annotators.

The datasets are described in detail and used in the experiments of papers below. Please cite one of them if you use this material in your research.

  * [How Naked is the Naked Truth? A Multilingual Lexicon of Nominal Compound Compositionality](http://aclweb.org/anthology/P16-2026)
  * [Predicting the Compositionality of Nominal Compounds: Giving Word Embeddings a Hard Time](http://aclweb.org/anthology/P16-1187)
  * [Filtering and Measuring the Intrinsic Quality of Human Compositionality Judgments](http://aclweb.org/anthology/W16-1804)
  
Our methodology is inspired from [Reddy, McCarthy and Manandhar](http://www.aclweb.org/anthology/I/I11/I11-1024). We include their set of 90 compounds and judgments in our dataset for the analyses in our papers. We do not include their dataset here, though. Please download their data and cite their paper if you use the full EN dataset.
  
  
## Folders

  * annotations: results, including raw files with individual annotations, averaged unfiltered and averaged filtered data (see filtering parameters below)
  * bin: scripts used to filter and estimate the quality of datasets (MWE workshop paper)
  * compounds-lists: contains the list of compounds and auxiliary information (gender, number, example sentences, etc) given to replace placeholders in MTurk questionnaire (in csv format for FR and EN) or used to create the dynamic HTML annotation interface (in MySQL database format for PT)
  * questionnaires: MTurk and HTML interfaces used in data collection. FR interface in HTML is included, but the data in this package comes from MTurk.

  
## Scripts

 * _Generation of unfiltered, averaged files used in ACL long and short papers_

`../bin/filter-answers.py --zscore-thresh=10000000 --spearman-thresh=-1 --batch-file en.raw.csv --lang en > en.unfiltered.csv 2> en.unfiltered.log`
`../bin/filter-answers.py --zscore-thresh=10000000 --spearman-thresh=-1 --batch-file fr.raw.csv --lang fr > fr.unfiltered.csv 2> fr.unfiltered.log`
`../bin/filter-answers.py --zscore-thresh=10000000 --spearman-thresh=-1 --batch-file pt.raw.csv --lang pt > pt.unfiltered.csv 2> pt.unfiltered.log`

 * _Generation of filtered averaged files used in MWE workshop paper_

`../bin/filter-answers.py --zscore-thresh=2.2 --spearman-thresh=0.5 --batch-file en.raw.csv --lang en > en.filtered.csv 2> en.filtered.log`
`../bin/filter-answers.py --zscore-thresh=2.5 --spearman-thresh=0.5 --batch-file fr.raw.csv --lang fr > fr.filtered.csv 2> fr.filtered.log`
`../bin/filter-answers.py --zscore-thresh=2.2 --spearman-thresh=0.5 --batch-file pt.raw.csv --lang pt > pt.filtered.csv 2> pt.filtered.log`

 * _Generation of graphics and evaluation of datasets_

`../bin/intrinsic-quality-dataset.py --avg-file en.unfiltered.csv 2> en.unfiltered.quality`
`../bin/intrinsic-quality-dataset.py --avg-file en.filtered.csv 2> en.filtered.quality`
`../bin/intrinsic-quality-dataset.py --avg-file fr.unfiltered.csv 2> fr.unfiltered.quality`
`../bin/intrinsic-quality-dataset.py --avg-file fr.filtered.csv 2> fr.filtered.quality`
`../bin/intrinsic-quality-dataset.py --avg-file pt.unfiltered.csv 2> pt.unfiltered.quality`
`../bin/intrinsic-quality-dataset.py --avg-file pt.filtered.csv 2> pt.filtered.quality`
`mkdir -p graphics`
`mv *.pdf graphics`

__Note__ : Data may differ slightly from papers because we added some new annotations since the papers were written.

