#!/usr/bin/python
#coding: utf-8

from __future__ import print_function
from __future__ import division
import sys
import re

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