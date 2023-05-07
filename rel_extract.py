import json
import pandas as pd
import numpy as np
# import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))

def write_output_file(relations, filename = "output.csv"):
    """The list of relations into a csv file for the evaluation script

    Args:
        relations (list(tuple(str, str))): a list of the relations to write
            the first element of the tuple is the PERSON, the second is the
            GeoPolitical Entity
        filename (str, optional): Where to write the output file. Defaults to "output.csv".
    """
    out = []
    for person, gpe in relations:
        out.append({"PERSON": person, "GPE": gpe})
    df = pd.DataFrame(out)
    df.to_csv(filename, index=False)

#  Build a training/validation/testing pipeline for relation extraction
#  then write the list of relations extracted from the *test set* to "output.csv"
#   using the write_output_file function.

sent_tokenizer = CountVectorizer()

# data-processing
def processingX(data, type='train'):
  x_matrix = []
  meaningful_sentences = []

  for item in data:
    flags = [0] * 4  # 0: PERSON 1: GPE 2: Distance between entities
    bow_start = 99999999
    bow_end = -1

    # Bag of words between M1 and M2
    for ent in item['entities']:
      if ent['label'] == 'PERSON':
        # check whether name is PROPN
        name_range = [i for i in range(ent['start'], ent['end']) if item['pos'][i] == 'PROPN' ]
        if len(name_range):
          flags[0] = 1

      if ent['label'] == 'GPE':
        flags[1] = 1

      bow_start = min(bow_start, ent['start'])
      bow_end = max(bow_end, ent['end'])

    flags[2] = bow_end - bow_start

    # Occurrence of PROPN in bag of words between M1 and M2
    flags[3] = sum([i == "PROPN" for i in item["pos"][bow_start:bow_end]])
    x_matrix.append(flags)

    # get meaning tokens and convert to sentences
    meaningful_toks = [item['tokens'][i].lower() for i in range(bow_start, bow_end) if not item['isstop'][i] and item['isalpha'][i] and item['shape'][i][0] == 'X']
    meaningful_sentences.append(' '.join(meaningful_toks))

  # Convert sentences to token counts
  # to confirm test data not seen during training
  if type == 'train':
    sent_tokenizer.fit(meaningful_sentences)
  
  X = sent_tokenizer.transform(meaningful_sentences)

  return np.concatenate((np.array(x_matrix), X.toarray()), axis=1)

# get labels Y
def processingY(data):
  labels = []
  for item in data:
      if item["relation"]["relation"] == "/people/person/nationality":
          labels.append(1)
      else:
          labels.append(0)
  return labels

X = processingX(train_data)
Y = processingY(train_data)

# split data
train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size = 0.3)

# grid search for best model
def searchBest(X, Y):
  param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 
   
  search = GridSearchCV(LogisticRegression(max_iter = 500), param_grid, refit = True, verbose = 3)
    
  # fitting the model for grid search
  search.fit(X, Y)
  return search.best_params_

best_parameters = searchBest(val_x, val_y)

# Train model
lr = LogisticRegression(C = best_parameters['C'], max_iter = 500)
lr_model = lr.fit(train_x, train_y)

# predict on validation
pred_y = lr_model.predict(val_x)
f1 = f1_score(val_y, pred_y)
print('F1 score:', f1)

# process test_data
test_X = processingX(test_data, 'test')

# predictions
pred_test = lr_model.predict(test_X)


# Remove this and write out the relations you extracted (obviously don't hard code them)
country_names = open("country_names.txt", "r").read()
countries = country_names.split('\n')

# get dependency relationships
def getDepRelation(data, element):
    dep_path=[]
    index=element['start']
    while data['dep'][index]!='ROOT':
        dep_path.append(index)
        index=data['dep_head'][index]
    dep_path.append(index)
    return dep_path

relations = []
for i in range(len(pred_test)):
    if pred_test[i] == 1:
        related_d = test_data[i]

        # get persons and countries in the sentence
        p_list = []
        c_list = []
        for entity in related_d["entities"]:
            if entity['label'] == 'PERSON':
                p_list.append(entity)
            if entity['label'] == 'GPE' and ' '.join(related_d["tokens"][entity['start']:entity['end']]) in countries:
                c_list.append(entity)

        # get the dependency relationships
        dep_person_curves = [ getDepRelation(related_d, i) for i in p_list ]
        dep_country_curves = [ getDepRelation(related_d, i) for i in c_list ]
        if not len(dep_person_curves) or not len(dep_country_curves):
            continue

        best_p = 0
        best_c = 0
        max_distance = 999999

        # find the best pairs
        for p in range(len(dep_person_curves)):
            for c in range(len(dep_country_curves)):
                c_indexes = dep_country_curves[c]
                p_indexes = dep_person_curves[p]
                
                sub_p = p_indexes[1:]
                sub_c = c_indexes[-len(p_indexes[1:]):]

                if sub_p == sub_c:
                    dist = len(set(c_indexes) - set(p_indexes)) + 1
                    if dist < max_distance:
                        best_p = p
                        best_c = c
                        max_distance = dist

        # get person and country, add the pair to relations
        person_name = " ".join(related_d["tokens"][p_list[best_p]['start'] : p_list[best_p]['end']])
        country_name = " ".join(related_d["tokens"][c_list[best_c]['start'] : c_list[best_c]['end']])
        
        relations.append((person_name, country_name))

write_output_file(relations)

