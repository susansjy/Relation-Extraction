
import json
import pandas as pd
import numpy as np
# import spacy
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score
from sklearn.model_selection import cross_val_score
# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))

# read in the data
train_data = json.load(open("sents_parsed_train.json", "r"))
test_data = json.load(open("sents_parsed_test.json", "r"))

def print_example(data, index):
    """Prints a single example from the dataset. Provided only
    as a way of showing how to access various fields in the
    training and testing data.

    Args:
        data (list(dict)): A list of dictionaries containing the examples 
        index (int): The index of the example to print out.
    """
    # NOTE: You may Delete this function if you wish, it is only provided as 
    #   an example of how to access the data.
    
    # print the sentence (as a list of tokens)
    print("Tokens:")
    print(data[index]["tokens"])

    # print the entities (position in the sentence and type of entity)
    print("Entities:")
    for entity in data[index]["entities"]:
        print("%d %d %s" % (entity["start"], entity["end"], entity["label"]))
    
    # print the relation in the sentence if this is the training data
    if "relation" in data[index]:
        print("Relation:")
        relation = data[index]["relation"]
        print("%d:%s %s %d:%s" % (relation["a_start"], relation["a"],
            relation["relation"], relation["b_start"], relation["b"]))
    else:
        print("Test examples do not have ground truth relations.")

def write_output_file(relations, filename = "q3.csv"):
    """The list of relations into a csv file for the evaluation script

    Args:
        relations (list(tuple(str, str))): a list of the relations to write
            the first element of the tuple is the PERSON, the second is the
            GeoPolitical Entity
        filename (str, optional): Where to write the output file. Defaults to "q3.csv".
    """
    out = []
    for person, gpe in relations:
        out.append({"PERSON": person, "GPE": gpe})
    df = pd.DataFrame(out)
    df.to_csv(filename, index=False)

# print a single training example
print("Training example:")
print_example(train_data, 1)

print("---------------")
print("Testing example:")
# print a single testing example
# the testing example does not have a ground
# truth relation
print_example(test_data, 2)

#TODO: build a training/validation/testing pipeline for relation extraction
#       then write the list of relations extracted from the *test set* to "q3.csv"
#       using the write_output_file function.

# data pre-processing
def tokenize(data):
    # clean sentences
    clean_sentences=[]
    # labels in the entities
    features=[]
    # each sample
    for cur in data:
        entities=cur["entities"]
        leftmostindex=10000
        rightmostindex=-1
        # Entities labels + The distance between the entities
        feature=[0,0,0]
        # find the range between entities
        for entity in entities:
            if entity['label']=='PERSON':
                feature[0]=1
            if entity['label']=='GPE':
                feature[1]=1
            if cur["pos"][entity['start']]=='PROPN' or  cur["pos"][entity['end']]=='PROPN':
                leftmostindex=min(leftmostindex,entity['start'])
                rightmostindex=max(rightmostindex,entity['end'])
        feature[2]=rightmostindex-leftmostindex
        # Pos the number of PROPN
        feature.append(sum([ i=="PROPN" for i in cur["pos"][leftmostindex:rightmostindex]]))
        assert len(feature)==4
        features.append(feature)
        # extract sentences between entities
        cur_sentence=''
        for index in range(leftmostindex,rightmostindex):
            if not cur["isstop"][index] and cur["isalpha"][index] and cur["shape"][index].startswith("X"):
                cur_sentence+=cur["tokens"][index].lower()+" "
        clean_sentences.append(cur_sentence)
    # Bags of words between entities
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(clean_sentences).toarray()
    vector=np.concatenate((vector,np.array(features)),axis=1)

    return np.array(vector),vectorizer

# get label; if exist nationality relation in the sentence
def get_label(data):
    labels=[]
    for cur in data:
        relation = cur["relation"]
        if relation["relation"]=="/people/person/nationality":
            labels.append(1)
        else:
            labels.append(0)
    return labels

# find the path from current node to the ROOT in the dependency tree
def pathfind(data,element):
    dep_path=[]
    index=element['start']
    while data['dep'][index]!='ROOT':
        dep_path.append(index)
        index=data['dep_head'][index]
    dep_path.append(index)
    return dep_path

# tokenize unseen data (testing data)
def tokenize_unseen(data,vectorizer):
    # clean sentences
    clean_sentences=[]
    # labels in the entities
    features=[]
    # each sample
    for cur in data:
        entities=cur["entities"]
        leftmostindex=10000
        rightmostindex=-1
        # Entities labels + The distance between the entities
        feature=[0,0,0]
        # find the range between entities
        for entity in entities:
            if entity['label']=='PERSON':
                feature[0]=1
            if entity['label']=='GPE':
                feature[1]=1
            if cur["pos"][entity['start']]=='PROPN' or  cur["pos"][entity['end']]=='PROPN':
                leftmostindex=min(leftmostindex,entity['start'])
                rightmostindex=max(rightmostindex,entity['end'])
        feature[2]=rightmostindex-leftmostindex
        # Pos: the number of PROPN
        feature.append(sum([ i=="PROPN" for i in cur["pos"][leftmostindex:rightmostindex]]))
        assert len(feature)==4
        features.append(feature)
        # extract sentences between entities
        cur_sentence=''
        for index in range(leftmostindex,rightmostindex):
            if not cur["isstop"][index] and cur["isalpha"][index] and cur["shape"][index].startswith("X"):
                cur_sentence+=cur["tokens"][index].lower()+" "
        clean_sentences.append(cur_sentence)
    # Bags of words between entities
    vector = vectorizer.transform(clean_sentences).toarray()
    vector=np.concatenate((vector,np.array(features)),axis=1)
    return np.array(vector)

# get Y
Y=get_label(train_data)
# get X and vectorizer
X,vectorizer=tokenize(train_data)
# split the data set
x_train, x_val, y_train, y_val = \
            train_test_split(X, Y, test_size=0.3,shuffle=True)
print("Finish data splitting")

# find the best hyperparameter setting via Cross-validation + grid search
# C=[0.0001,0.001,0.01,0.1,1,10,100,1000]
# max_iters=[100,500,1000]
# for c in C:
#     for max_iter in max_iters:
#         model = LogisticRegression(max_iter=max_iter,C=c,class_weight='balanced')
#         scores = cross_val_score(model, X, Y, cv=5)
#         print("current c is ",c,"current iteration is",max_iter, "score is ",scores.mean())

# Training
# The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
model = LogisticRegression(max_iter=100,C=0.1,class_weight='balanced')
model.fit(x_train,y_train)
print("Finish training")
# validation
pred_val = model.predict(x_val)
precision=precision_score(y_val,pred_val)
acu = accuracy_score(y_val, pred_val)
recall = recall_score(y_val, pred_val, average="macro")
score=f1_score(y_val, pred_val)
print("precision %f "%precision)
print("recall %f "%recall)
print("auccuacy %f "%acu)
print("score %f "%score)

# get the relations from the test data
unseen_data=tokenize_unseen(test_data,vectorizer)
# get countries name for validation purpose
content = open("country_names.txt", "r").read()
country_names = content.split("\n")

pred_unseen = model.predict(unseen_data)

# Example only: write out some relations to the output file
# normally you would use the list of relations output by your model
# as an example we have hard coded some relations from the training set to write to the output file
# TODO: remove this and write out the relations you extracted (obviously don't hard code them)
# relation extraction
relations = []
for i in range(len(pred_unseen)):
    # only extract relation from sentenses with label 1
    if pred_unseen[i] == 1:
        cur = test_data[i]
        # record person entity and coutry entity in the sentence
        persons = []
        countries = []
        for entity in cur["entities"]:
            if entity['label'] == 'PERSON':
                persons.append(entity)
            if entity['label'] == 'GPE' and ' '.join(cur["tokens"][entity['start']:entity['end']]) in country_names:
                countries.append(entity)
        # record the path from current person/country to the ROOT in the dependency tree
        dep_path_person = []
        dep_path_country = []
        for person in persons:
            path = pathfind(cur, person)
            dep_path_person.append(path)
        for country in countries:
            path = pathfind(cur, country)
            dep_path_country.append(path)
        if len(dep_path_person) <= 0 or len(dep_path_country) <= 0:
            continue
        most_match_p = 0
        most_match_c = 0
        maxdistance = 100000
        for p in range(len(dep_path_person)):
            for c in range(len(dep_path_country)):
                country = dep_path_country[c]
                person = dep_path_person[p]
                if person[1:] == country[-len(person[1:]):]:
                    tree_distance = len(set(country) - set(person)) + 1
                    if tree_distance < maxdistance:
                        most_match_p = p
                        most_match_c = c
                        maxdistance = tree_distance
        relations.append((" ".join(cur["tokens"][persons[most_match_p]['start']:persons[most_match_p]['end']]),
                          " ".join(cur["tokens"][countries[most_match_c]['start']:countries[most_match_c]['end']])))

write_output_file(relations)