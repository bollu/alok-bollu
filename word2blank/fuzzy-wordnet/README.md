# Building and running `graph.out`:

- We first need a compiled version of `igraph`: follow the following steps, **starting from the current folder**:
```
$ unzip igraph-0.8.2.zip
$ cd igraph-0.8.2
$ ./bootstrap.sh
$ ./configure --prefix=$HOME/.local # installs stuff into your home folder
$ make -j # make -j: build with full parallelism
$ make install # installs the stuff into ~/.local/lib and ~/.local/bin
```

Then run:

```
$ make
$ mkdir -p logs
$ ./graph.out <path-to-fasttext-dump> 10000 &> logs/fasttext-wiki-news-300-scc-vocab=10000.txt
```
