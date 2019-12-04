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
