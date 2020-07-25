from random import shuffle
from tqdm import tqdm

# st = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

filename = input('Please enter file name: ')
window = input('Please enter window size: ')

# filename = './text0'
# window = 4

sentence = []
rewrite = []
ix = 0
i = 0

with open(filename, 'r') as f:
    lines = f.read()
    words = lines.split()
    for ix in tqdm(range(len(words))):
        sentence.append(words[ix])
        if ix % int(window) == 0:
            # print(sentence)
            rewrite.append(sentence)
            sentence = [words[ix]]
        ix += 1

g = open('./shuffled.txt', 'w')
rewrite = rewrite[1:]
shuffle(rewrite)
for i in range(len(rewrite)):
    for j in range(len(rewrite[i])):
        g.write(rewrite[i][j] + ' ')
g.write('\b\n')
g.close()
