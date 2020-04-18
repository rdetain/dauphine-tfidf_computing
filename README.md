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

## Using Hadoop Streaming

For this part, I created 2 different scripts : one **mapper** and one **reducer**.

Mapper

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
