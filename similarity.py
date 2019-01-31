# Check similarity between two text files
# Author: Bahar ali
# January 14, 2019

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np 
import nltk 
import csv
import glob
import os
from sklearn.metrics import jaccard_similarity_score
from math import*

def jaccard_similarity(x,y):
	intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
	union_cardinality = len(set.union(*[set(x), set(y)]))
	return intersection_cardinality/float(union_cardinality)
	
def cos_sim(a, b): 
	dot_product = np.dot(a, b) 
	norm_a = np.linalg.norm(a) 
	norm_b = np.linalg.norm(b) 
	return dot_product / (norm_a * norm_b) 
	
def getSimilarity(dict1, dict2): 
	all_words_list= [] 
	for key in dict1: 
		all_words_list.append(key) 
	for key in dict2: 
		all_words_list.append(key) 
	all_words_list_size = len(all_words_list) 
	v1 = np.zeros(all_words_list_size, dtype=np.int)
	v2 = np.zeros(all_words_list_size, dtype=np.int) 
	i = 0 
	for (key) in all_words_list: 
		v1[i] = dict1.get(key, 0) 
		v2[i] = dict2.get(key, 0) 
		i = i + 1 
	return cos_sim(v1, v2); 

def process(file): 
	raw = open(file).read() 
	tokens = word_tokenize(raw) 
	words = [w.lower() for w in tokens] 
	porter = nltk.PorterStemmer ()
	stemmed_tokens = [porter.stem(t) for t in words] 
	# Removing stop words 
	stop_words = set(stopwords.words('english')) 
	filtered_tokens = [w for w in stemmed_tokens if not w in stop_words] 
	# count words 
	count = nltk.defaultdict(int)
	for word in filtered_tokens: 
		count[word] += 1 
	return count; 
 
	
myfile = open('similarity_cosine.csv', mode='w')
myfile_writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
myfile_writer.writerow(['File 1', 'File 2', 'Similarity'])
 
myfile2 = open('similarity_jcard.csv', mode='w')
myfile_writer2 = csv.writer(myfile2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
myfile_writer2.writerow(['File 1', 'File 2', 'Similarity'])

for filex in os.listdir('files'):
	dict1 = process('files/'+filex) 
	for filey in os.listdir('files'):
		dict2 = process('files/'+filey) 
		sim = getSimilarity(dict1,dict2)
		sim2 = jaccard_similarity(dict1,dict2)
		print('similarity between ' + filex + ' & '+ filey +' is '+ str(sim2))
		myfile_writer.writerow([filex, filey, sim])
		myfile_writer2.writerow([filex, filey, sim2])
   
 