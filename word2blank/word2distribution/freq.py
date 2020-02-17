import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import struct

with open("models/text0", "rb") as f:
    lines = f.read().split(b'\n')
    word2vec = {}
    word2len = {}
    [vobabsize, embedsize] = map(int, lines[0].decode("utf-8").split())

    for l in lines[1:]:
        ix = l.find(b' ')
        if ix == -1: continue
        w = l[:ix]
        try:
            d = l[ix+1:]
            d = np.frombuffer(d, dtype=np.float32, count=embedsize)
            word2vec[w] = d

            norm = np.linalg.norm(d)
            word2len[w] = norm
        except Exception as e:
            print(e)
            word2vec[w] = np.zeros(embedsize, dtype=np.float32)
            word2len[w] = 0

def cosine(focus):
    print("focus: %s" % focus)
    w2v = word2vec.items()
    w2v.sort(key=lambda kv: kv[1].dot(focus), reverse=True)
    print("------------")
    print("\n".join([ "%20s %4.2f" % (kv[0], kv[1].dot(focus)) for kv in w2v[:20]]))

with open("text0", "r") as corpus:
    word2freq = {}
    for l in corpus:
        for w in  l.split(' '):
            if w in word2freq:
                word2freq[w] += 1
            else:
                word2freq[w] = 1

    w2f = word2freq.items()
    w2f.sort(key=lambda kv: kv[1], reverse=True)
    print("\n".join(["%20s : %s" % (kv[0], kv[1]) for kv in w2f[:20]]))
    medianf = w2f[len(w2f) // 2]

cosine(word2vec['apollo'])

for w in word2freq.keys():
    if w not in word2len: continue
    if word2len[w] == 0: continue
    if word2freq[w] > 1000: continue
    plt.scatter(word2len[w], word2freq[w])

plt.show()

