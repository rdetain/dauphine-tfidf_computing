#!/usr/bin/python
#coding: utf-8

from __future__ import print_function
from __future__ import division
import sys
import csv
from math import log10
from collections import defaultdict

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
	# **OUTPUT DATA*** | ((word, doc_ID), TF-IDF) on each line of stdout
	key_formated = '{:_<30}'.format("%s,%i" % (word,docid))
	print("%s\t%i\t%i\t%i\t%.*f" % (key_formated,wordcount,wordcount_per_doc,df_t,5,TFIDF))


# TEST Results - TOP 20 tf-idf for each document
for docid in docid_list:
    words_top20_tfidf = sorted([word_block for word_block in words if word_block[0][1] == docid], key=lambda x: x[4], reverse=True)[:20]
    print(words_top20_tfidf)