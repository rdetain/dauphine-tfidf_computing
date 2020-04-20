# dauphine-tfidf_computing

Author : Rudy Detain  

University : Paris Dauphine

## Purpose
Considering the problem of calculating the TF-IDF score for each (word, document) couple in a set of documents, I performed an experimental analysis in order to compare performances (execution time) of 3 different methods :

- tf-idf calculation using scikitlearn library,
- tf-idf calculation using Hadoop Streaming,
- tf-idf calculation using Spark.

A basic NLP pre-processing has been performed within each of these methods :

- words tokenization,
- stop-words removing.

## Input
I used a set of 54 documents. Each document is a chapter of JRR Tolkien's *Lord of the Rings*. 


## Using Scikitlearn
Stop-words list creation
```
stop_words = list(set(stopwords.words('english'))) + list ({',','.','-',';',':','(',')','?','-PRON-','!',' '})
df = pd.DataFrame(stop_words) 
df.to_csv('./book/stop_words.txt', sep='\t', index=False)
```
Tokenizer creation
```
tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
```

Loading spacy model used for lemmatization
```
nlp = spacy.load('en_core_web_sm')
```

Iterating NLP process on each document
```
%time 
listdoc = [] 
for i in range(0,54): 
  text_file = open("book/{}.txt".format(i), "r", encoding="ISO-8859-1") 
  lines = text_file.read() 
  text_file.close() 
  listtoken = ' '.join([item for item in tokenizer.tokenize(lines.lower().replace("'"," ")) if item not in stop_words]) 
  lemma = nlp(listtoken) 
  listlemma = [] 
  for token in lemma: 
    listlemma.append(token.lemma_) 
  listdoc.append(listlemma)
```

TF-IDF calculation
```
%time 
def dummy_fun(doc): 
  return doc 
  
tfidf_vectorizer = TfidfVectorizer(
  analyzer='word',
  tokenizer=dummy_fun,
  preprocessor=dummy_fun,
  token_pattern=None)
  
tfidf = tfidf_vectorizer.fit_transform(listdoc)
```
Results

Execution time for this script was **52 sec**. It has been performed using a Macbook Pro with the tech specs below :

- 2,4 GHz Intel Core i5
- 8 Go 1333 MHz DDR3

## Using Hadoop Streaming

For this part, I created 2 different scripts : one **mapper** and one **reducer**.

Both scripts are commented here below.

*Mapper*

```
words=[]
doc_id=0
wordcount = 1
wordcount_per_doc = 0
df_t=1

#load each file content in the stdin output
for line in sys.stdin:

	# remove spaces
	line = line.strip()

	# assign ID for each read document 
	if line.isdigit() == True:
		doc_id = int(line)

	# word tokenize
	words_in_line = line.split()

	# format each word
	words_in_line = [word.lower() for word in words_in_line]# lower case
	words_in_line = [re.sub(r'[^\w]', '', word) for word in words_in_line] # remove punctuation and special characters
	stopwords=[] # filtering stop words
	for line in open('stopwords_en.txt'):
		stopwords.append(line.strip())
	words_in_line = [word for word in words_in_line if word not in stopwords]
	words_in_line = [word for word in words_in_line if len(word)>2] #remove words with less than 3 characters

	# add words from the current line to the final words list
	words += words_in_line

	# calculate the wordcount_per_doc by adding the wordcount of each line
	wordcount_per_doc += len(words_in_line)

# 5. ***OUTPUT DATA*** | each word is returned to the stdout
for word in words:
	print("%s,%i\t%i\t%i\t%i" % (word,doc_id,wordcount,wordcount_per_doc, df_t))

```

*Reducer*

```
words = []										# words list
last_word_docid_pair = None									# for wordcount calculation
df_t_dict = defaultdict(lambda: set())	# for df_t calculation (for a word 'x', df_t is the number of documents containing 'x')
docid_list = set()										# number of documents

for line in sys.stdin:
	# get key (word, docid) and values (wordcount, wordcount_per_doc, df_t) from the stdout of the mapper
	line = line.strip()
	key,wordcount,wordcount_per_doc,df_t = line.split("\t")
	wordcount_per_doc=int(wordcount_per_doc)
	wordcount = int(wordcount)
	df_t = int(df_t)
	word,docid = key.split(",")
	docid = int(docid)
	word_docid_pair = (word,docid)
	# wordcount calculation
	if last_word_docid_pair is None:						# 1st word treatment
		last_word_docid_pair = word_docid_pair
		last_wordcount = 0
		last_wordcount_per_doc = wordcount_per_doc
		last_df_t = df_t
	if word_docid_pair == last_word_docid_pair:
		last_wordcount += wordcount
	else:
		words.append([last_word_docid_pair,last_wordcount,last_wordcount_per_doc,last_df_t])
		# set new values
		last_word_docid_pair = word_docid_pair
		last_wordcount = wordcount
		last_wordcount_per_doc = wordcount_per_doc
		last_df_t = df_t
	# update the list of documents containing 'word'
	dic_value = df_t_dict[word]
	dic_value.add(docid)
	df_t_dict[word] = dic_value
	# update the list of documents
	docid_list.add(docid)

# add the last word which has not been treated during the previous step
words.append([last_word_docid_pair,last_wordcount,last_wordcount_per_doc,last_df_t])
# final number of documents calculation
N = len(docid_list)

for word_block in words:
	word,docid,wordcount,wordcount_per_doc,df_t = word_block[0][0],int(word_block[0][1]),int(word_block[1]),int(word_block[2]),int(word_block[3])
	# return, for each word, the final df_t
	df_t = len(df_t_dict[word])
	# TF-IDF calculation = wordcount x wordcount_per_doc x log10(N/df_t)
	word_block.append(wordcount * wordcount_per_doc * log10(N/df_t))
	TFIDF = word_block[4]
	# ***OUTPUT DATA*** | ((word, doc_ID), TF-IDF) on each line of stdout
	key_formated = '{:_<30}'.format("%s,%i" % (word,docid))
	print("%s\t%i\t%i\t%i\t%.*f" % (key_formated,wordcount,wordcount_per_doc,df_t,5,TFIDF))


# TEST Results - TOP 20 tf-idf for each document
for docid in docid_list:
    words_top20_tfidf = sorted([word_block for word_block in words if word_block[0][1] == docid], key=lambda x: x[4], reverse=True)[:20]
    print(words_top20_tfidf)
```

Results

Execution time for both scripts was **27 sec**, using Hadoop Streaming through the cluster of Paris Dauphine University.

## Using Hadoop Streaming

PySpark Script

Here above the steps followed within the PySpark script I built :

- Load txt files into a dataframe
- Tokenize words of each document
- Term frequency in each document
- Term frequency in the whole corpus of documents
- TF-IDF calculation for each word
- Sort results by score

```
import numpy as np
from __future__ import division
from pyspark.sql import SQLContext, Row

## Load txt files into a dataframe

lordofthering = sc.wholeTextFiles("/FileStore/tables/lordofthering/").map(lambda x : (x[0].replace('dbfs:/FileStore/tables/lordofthering/',''), x[1]) ).toDF(["chapter","text"])

## Tokenize words of each document

from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
  
# tokenize text to words.
import re
def tokenize(s):
  return re.split("\\W+", s.lower())

tokenized_text = lordofthering.rdd.map(lambda x : (x[0], tokenize(x[1])) )
print(tokenized_text.take(1))

Beginning of the list obtained :
[('0.txt', ['prologue', '1', 'concerning', 'hobbits', 'this', 'book', 'is', 'largely', 'concerned', 'with', 'hobbits', 'and', 'from', 'its', 'pages', 'a', 'reader', 'may', 'discover', 'much', 'of', 'their', 'character', 'and', 'a', 'little', 'of', 'their', 'history', 'further', 'information', 'will', 'also', 'be', 'found', 'in', 'the', 'selection', 'from', 'the', 'red', 'book', 'of', 'westmarch'

## TERM FREQUENCY
# term frequencies in each document
term_frequency = tokenized_text.flatMapValues(lambda x: x).countByValue()
print(take(10, term_frequency.items()))

[(('52.txt', 'pelennor'), 1), (('39.txt', 'silence'), 2), (('54.txt', 'found'), 2), (('48.txt', 'riders'), 2), (('44.txt', 'gurgling'), 1), (('34.txt', 'pursuing'), 1), (('29.txt', 'eighteen'), 2), (('19.txt', 'companions'), 10), (('17.txt', 'mortals'), 1), (('40.txt', 'man'), 31)] CPU times: user 104 ms, sys: 12.1 ms, total: 117 ms Wall time: 1.65 s


## DOCUMENT FREQUENCY
#count how many documents a word appears in.
document_frequency = tokenized_text.flatMapValues(lambda x: x).distinct()\
                        .filter(lambda x: x[1] != '')\
                        .map(lambda x: (x[1],x[0])).countByKey()
document_frequency.items()
take(10, document_frequency.items())

## TF-IDF

def tf_idf(N, tf, df):
    result = []
    for key, value in tf.items():
        doc = key[0]
        term = key[1]
        df = document_frequency[term]
        if (df>0):
          tf_idf = float(value)*np.log(N/df)
        
        result.append({"doc":doc, "term":term, "score":tf_idf})
    return result

tf_idf_output = tf_idf(number_of_chapter, term_frequency, document_frequency)

print("{:20s} {:20s} {:15s}".format("Chapter", "Word", "TF-IDF Score")) 
print("{:20s} {:20s} {:15s}".format("----------------", "----------------", "----------------")) 
[print("{:20s} {:20s} {:3.6f}".format(item['doc'],item['term'],item['score'])) for item in tf_idf_output[0:10]]

Chapter Word TF-IDF Score 
---------------- ---------------- ---------------- 
52.txt pelennor 1.810109 
39.txt silence 0.401341 
54.txt found 0.074083 
48.txt riders 0.792831 
44.txt gurgling 1.810109 
34.txt pursuing 2.061423 
29.txt eighteen 5.242078 
19.txt companions 4.519851 
17.txt mortals 2.908721 
40.txt man 8.359571 

## Sort results by max score
print("{:20s} {:20s} {:15s}".format("Chapter", "Word", "TF-IDF Score")) 
print("{:20s} {:20s} {:15s}".format("----------------", "----------------", "----------------")) 
[print("{:20s} {:20s} {:3.6f}".format(item['doc'],item['term'],item['score'])) for item in sorted(tf_idf_output, key = lambda i: i['score'] ,reverse=True) [0:10]]

Chapter Word TF-IDF Score 
---------------- ---------------- ---------------- 
20.txt ugl 154.162208 
21.txt treebeard 146.361036 
20.txt k 138.915058 
4.txt ruffians 127.983719 
21.txt ents 104.986299 
20.txt grishn 
99.425580 20.txt kh 99.425580 
4.txt cotton 98.896510 
7.txt tom 88.300739 
30.txt gollum 78.904871 

```

Results

Execution time for this script was **5.41 sec**, using Databricks.

## Final Results

