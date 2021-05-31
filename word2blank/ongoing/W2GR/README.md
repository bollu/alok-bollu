#Grassmannian Text Embeddings

## Run the Code

We provide a shell script ``run.sh`` for compiling the source file and training embedding.

**Note: When preparing the training text corpus, make sure each line in the file is one document/paragraph.**

### Hyperparameters

Invoke the command without arguments for a list of hyperparameters and their meanings:
```
$ ./src/jose
Parameters:
        -train <file> (mandatory argument)
                Use text data from <file> to train the model
        -word-output <file>
                Use <file> to save the resulting word matrices
        -context-output <file>
                Use <file> to save the resulting word context matrices
        -doc-output <file>
                Use <file> to save the resulting document matrices
        -size <int>
                Set size of word vectors; default is 100
        -p <int>
		Set the paragraph matrices as p dimensional subspaces
	-l <int>
     		Set the word matrices as l dimensional subspaces
        -window <int>
                Set max skip length between words; default is 5
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the
                training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-3)
        -negative <int>
                Number of negative examples; default is 2
        -threads <int>
                Use <int> threads; default is 20
        -margin <float>
                Margin used in loss function to separate positive samples from negative samples; default is 0.15
        -iter <int>
                Run more training iterations; default is 10
        -min-count <int>
                This will discard words that appear less than <int> times; default is 5
	-binary <0/1>
		This will determine in what format the word matrices will be stored in the file
        -alpha <float>
                Set the starting learning rate; default is 0.04
        -debug <int>
                Set the debug mode (default = 2 = more info during training)
        -save-vocab <file>
                The vocabulary will be saved to <file>
        -read-vocab <file>
                The vocabulary will be read from <file>, not constructed from the training data
        -load-emb <file>
                The pretrained embeddings will be read from <file>

Examples:
./jose -train text.txt -word-output jose.txt -size 100 -margin 0.15 -window 5 -sample 1e-3 -negative 2 -iter 10
```

## Word Similarity Evaluation

Word similarity evaluation is done on the wikipedia dump with the file ``grass_sim.py``. A batch script will soon be written which will first download a zipped file of the pre-processed wikipedia dump (retrieved 2019.05; the zipped version is of ~4GB; the unzipped one is of ~13GB; for a detailed description of the dataset, see [its README file](datasets/wiki/README.md)), and then run our code on it. Finally, the trained embeddings are evaluated on three benchmark word similarity datasets: WordSim-353, MEN and SimLex-999.

## Sentiment Analysis Evaluation

``svm_classfication.py`` performs Grassmannian Kernel SVM on the document embeddings trained on IMDB 50k movie review dataset.

## Topical Document Classification Evaluation

``grass_classification.py`` performs KNN classification following the original 20 Newsgroup train/test split with the trained document embeddings as features.
