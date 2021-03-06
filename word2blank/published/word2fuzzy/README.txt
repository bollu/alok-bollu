Tools for computing distributed representtion of words
------------------------------------------------------

We provide an implementation of the Continuous Bag-of-Words (CBOW) and the Skip-gram model (SG), as well as several demo scripts.

Given a text corpus, the word2vec tool learns a vector for every word in the vocabulary using the Continuous
Bag-of-Words or the Skip-Gram neural network architectures. The user should to specify the following:
 - desired vector dimensionality
 - the size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model
 - training algorithm: hierarchical softmax and / or negative sampling
 - threshold for downsampling the frequent words 
 - number of threads to use
 - the format of the output word vector file (text or binary)

Usually, the other hyper-parameters such as the learning rate do not need to be tuned for different training sets. 

The script demo-word.sh downloads a small (100MB) text corpus from the web, and trains a small word vector model. After the training
is finished, the user can interactively explore the similarity of the words.

More information about the scripts is provided at https://code.google.com/p/word2vec/


- king man king -> woman word analogy!

## discrete / fuzzy intersections

Enter three words (EXIT to break): binary search

Word: binary  Position in vocabulary: 1996

Word: search  Position in vocabulary: 1460

                                              Word              Distance
------------------------------------------------------------------------
                                      enumerations		0.629941
                                          chapront		0.629188
                                            struct		0.622532
                                               xls		0.617342
                                        journaling		0.615803
                                         substring		0.615801
                                            queues		0.614423
                                    decompositions		0.613572
                                       executables		0.612928
                                           freedos		0.612928
                                           elgamal		0.609285

Enter three words (EXIT to break): gauge theory

Word: gauge  Position in vocabulary: 3816

Word: theory  Position in vocabulary: 209

                                              Word              Distance
------------------------------------------------------------------------
                                    chromodynamics		0.613065
                                         spacetime		0.599132
                                    supersymmetric		0.598432
                                          mordehai		0.595329
                                        invariance		0.594088
                                             lambs		0.588118
                                          monopole		0.583957
                                                mm		0.579943
                                               efe		0.579943
                                          theories		0.579489
                                           bourdon		0.579489
                                               qcd		0.578874
                                            kaluza		0.574663
                                          graviton		0.571662

Enter three words (EXIT to break): miles davis

Word: miles  Position in vocabulary: 1179

Word: davis  Position in vocabulary: 2393

                                              Word              Distance
------------------------------------------------------------------------
                                            herbie		0.625979
                                           ornette		0.624061
                                           sideman		0.617140
                                            guitar		0.610690
                                           sidemen		0.609306
                                           hancock		0.606479
                                          bassists		0.606479
                                            elmore		0.605645
                                       cranberries		0.602267


Enter three words (EXIT to break): steampunk cyberpunk

Word: steampunk  Position in vocabulary: 37853

Word: cyberpunk  Position in vocabulary: 8427

                                              Word              Distance
------------------------------------------------------------------------
                                         mechanist		0.708405
                                       cliffhanger		0.686113
                                       neuromancer		0.669031


- woman child => pregnant
- apollo: gives NASA. (and apllo greek) gives gods.
- (and ballistic training) == ballistic

- (and computer vision) -> SIGGRAPH.

- man rough woman -> word2vec: smooth. Us: no smooth, just random stuff.

- joke rule pun -> we give random answers that have high entropy. They still give "answers"". Our model is more "robust", in that
    it fails gracefully instead of always trying to produce output.

- (and secretary general)
- pow good 1.5
- pow bad 1.5
- pow sweet
- pow sour

- reldist ((discrete politics) colors) (KL)

                                       multiracial		0.005027
                                            visegr		0.005027
                                              njem		0.005027
                                        afrikaners		0.005027
                                            ffffff		0.005027
                                      chromaticity		0.005027
                                          plotdata		0.005027
                                              oara		0.005027
                                            region		0.005027
                                          fashions		0.005027
                                              wftu		0.005027
                                          karabakh		0.005027
                                            boomed		0.005027

