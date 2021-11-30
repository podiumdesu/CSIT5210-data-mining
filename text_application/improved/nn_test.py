from __future__ import division
from __future__ import print_function
from sentence_transformers import SentenceTransformer

from nltk import tokenize
import sys
import numpy as np

from pyod.models.copod import COPOD
from pyod.utils.data import evaluate_print

# 提取dataset中的评论并且
import csv

import nltk
nltk.download('punkt')


data = []
# with open('./paper.json', 'r') as file:
#     papers = json.load(file)
#     data = [paper["title"] for paper in papers]

with open('./papers.csv', 'rt',encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx > 601:
            break
        if idx > 101:
            data.append(row[4])

print("end csv reading")

data.append("Recurrent topic-transition gan for visual paragraph generation.")
data.append("Modelling one‐and two‐dimensional solid‐state NMR spectra ")
data.append("Storygan: A sequential conditional gan for story visualization")
data.append("Attngan: Fine-grained text to image generation with attentional generative adversarial networks")
data.append("A perpendicular-anisotropy CoFeB–MgO magnetic tunnel junction")
data.append("Tidal ventilation at low airway pressures can augment lung injury.")

data_vecs = []
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
for i, s in enumerate(data):
    sys.stdout.write("\r%d/%d" % (i, len(data)))
    sys.stdout.flush()
    r = model.encode(tokenize.sent_tokenize(s)).mean(axis=0)
    data_vecs.append(r.tolist())

print("end model encoding")

data_vecs = np.array(data_vecs)
print(np.shape(data_vecs))

clf = COPOD()
clf.fit(data_vecs)

print(np.shape(clf.labels_))

res = np.c_[clf.labels_, clf.decision_scores_].tolist()
for i, r in enumerate(res):
    r.append(data[i])

with open("new.csv", "w+",encoding = 'utf-8') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(res)
