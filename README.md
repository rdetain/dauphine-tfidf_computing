# dauphine-tfidf_computing

Author : Rudy Detain  

University : Paris Dauphine

## Purpose
Considering the problem of calculating the TF-IDF score for each (word, document) couple in a set of documents, I performed an experimental analysis in order to compare performances (execution time) of 3 different methods :

- tf-idf calculation using scikitlearn library,
- tf-idf calculation using MapReduce Hadoop Streaming,
- tf-idf calculation using Spark.

A basic NLP pre-processing has been performed within each of these methods :

- words tokenization,
- stop-words removing.

## Input
I used a set of 54 documents. Each document is a chapter of JRR Tolkien's *Lord of the Rings*. 

