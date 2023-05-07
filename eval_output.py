import numpy as np
import pandas as pd
import argparse
import json
import sys


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trainset", action="store_true")
args = ap.parse_args()

data = pd.read_csv("q3.csv")
if args.trainset:
    data_test = json.load(open("sents_parsed_train.json", "r"))
else:
    data_test = json.load(open("sents_parsed_test.json", "r"))

# read in the ground truth training data
rel = set()
for d in data_test:
    if "relation" not in d:
        print("Error: No ground truth relations in this dataset!")
        print("Note: relation are not provided for the test set (only the examiners have access to these relations)")
        print("You can run with the --trainset flag to evaluate against the training set.")
        sys.exit(0)

    # ignore relations that are not nationality
    if d["relation"]["relation"] != "/people/person/nationality":
        continue
    
    # get the person and the gpe joined as a string and in lower case
    person = ' '.join(d["relation"]["a"]).lower()
    gpe = ' '.join(d["relation"]["b"]).lower()
    rel.add((person, gpe))

# load your dataset and remove duplicates
dset = set()
for r in data.iterrows():
    # get the person and the gpe lowercase
    person = r[1]['PERSON'].lower()
    gpe = r[1]['GPE'].lower()
    dset.add((person,gpe))

# calculate the true positives and false positives
tp = 0
fp = 0
for p,g in dset:
    # is an identified relation in the ground truth
    if (p, g) in rel:
        tp += 1
    else:
        fp += 1

precision = tp/(tp+fp)
recall = tp/len(rel)
print('True positives:', tp)
print('False positives:', fp)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', 2 / (1/recall + 1/precision))

