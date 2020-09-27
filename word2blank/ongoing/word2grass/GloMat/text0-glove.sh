#!/bin/bash
set -e
set -o xtrace

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

rm build/glove build/glove.o build/common.o || true
make

CORPUS=../../../utilities/text0
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=matrices
ETA=0.1
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=20
MAX_ITER=3
# WINDOW_SIZE=15
WINDOW_SIZE=4
WRITE_HEADER=1
BINARY=1
NUM_THREADS=1
X_MAX=10
if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi
# if [ ! -e text8 ]; then
#   if hash wget 2>/dev/null; then
#     wget http://mattmahoney.net/dc/text8.zip
#   else
#     curl -O http://mattmahoney.net/dc/text8.zip
#   fi
#   unzip text8.zip
#   rm text8.zip
# fi

echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
# gdb -ex run --args $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -write-header $WRITE_HEADER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -write-header $WRITE_HEADER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -eta $ETA
