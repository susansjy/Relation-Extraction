# Relation-Extraction
A natural language processing project that involves designing and implementing a relation extraction model to extract the relation 'nationality' between a person and a geopolitical entity (GPE) from a dataset of text documents. The goal of the task is to develop a good training/validation pipeline and to explore different lexical, syntactic or semantic features to extract all relations of the type nationality that exist in the test data.

## Data
The data is provided in two files: train.json and test.json, which contain sentences in JSON format. Each sentence is represented by a list of tokens, and each token is represented by a dictionary with the following fields:
- "word": the word form of the token
- "lemma": the lemma of the token
- "pos": the part-of-speech tag of the token
- "ner": the named entity recognition tag of the token
- "dep": the dependency relation of the token to its parent in the parse tree
- "head": the index of the parent token in the sentence

Additionally, a file containing a list of country names (country_names.txt) is provided for use in the task.

## Implementation
The relation extraction algorithm is based on the "Feature-based supervised relation classifiers" technique introduced in the Speech and Language Processing textbook. The implementation uses scikit-learn's LogisticRegression classifier and incorporates various lexical, syntactic, and semantic features, including:
- the words between the two entities
- the part-of-speech tags of the words between the two entities
- the named entity recognition tags of the words between the two entities
- the dependency paths between the two entities
- the shortest dependency path between the two entities
- the presence of country names in the sentence

## Output Format
The output of the rel_extract.py script is a CSV file named q3.csv, which contains a list of all the relations of the type nationality that exist in the test data. Each row in the CSV file represents a relation and has the following format: person_name, country_name, where person_name is the name of the person in the relation and country_name is the name of the country in the relation.

## Evaluation
To evaluate the performance of the extracted relations, run the eval_output.py script with the --trainset flag to evaluate against the ground truth in the training data. The script calculates the precision, recall, and F1 score of the extracted relations.

## Conclusion
This task demonstrates the use of feature-based supervised classifiers for relation extraction in natural language processing. By incorporating various lexical, syntactic, and semantic features, the algorithm is able to accurately extract relations of the type nationality from text documents.
