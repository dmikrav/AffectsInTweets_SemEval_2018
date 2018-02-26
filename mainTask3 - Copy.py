# -*- coding: utf-8 -*-
# encoding=utf8
import sys, json, re
import numpy as np
from vaderSentiment import vaderSentiment as vs
analyzer = vs.SentimentIntensityAnalyzer()
reload(sys)
sys.setdefaultencoding('utf8')
affect_list = ["valence"]
import os
cwd = os.getcwd()
print cwd
with open(os.path.join(cwd, 'dataset', 'task3', 'train', 'dataset_json_train.txt')) as data_file:
    train_data = json.load(data_file)
with open(os.path.join(cwd, 'dataset', 'task3', 'dev', 'dataset_json_dev.txt')) as data_file:
    development_data = json.load(data_file)
with open(os.path.join(cwd, 'dataset', 'task3', 'test', 'dataset_json_test.txt')) as data_file:
    test_data = json.load(data_file)

test_data = development_data

import sklearn.ensemble, sklearn.metrics#, sklearn.cross_validation
from sklearn.metrics import mean_squared_error, r2_score
import scipy
import math
import numpy as np
import time

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

######################################################################################

dict_bag_of_words = {}
dict_curr_number = 0

def bag_of_words(word):
    global dict_curr_number
    if not word in dict_bag_of_words:
        dict_bag_of_words[word] = dict_curr_number
        dict_curr_number += 1
for affect in affect_list:
    for idd in train_data[affect]:
        tmp = train_data[affect][idd]["atext_clear_lower"].split(" ")
        for word in tmp:
            bag_of_words(word)
n_bag_of_words = len(dict_bag_of_words)
print n_bag_of_words
def make_vector_from_bag_of_words(sentence):
    tmp_list = sentence.split(" ")
    res = [0] * n_bag_of_words
    for word in tmp_list:
        res[dict_bag_of_words[word]] += 1
    return res


##########################################################################################




with open(os.path.join(cwd, 'DepecheMood_V1.0', 'DepecheMood_freq.txt')) as data_file:
    depecheMood = data_file.readlines()
depecheMood = depecheMood[1:]

import re
dict_depecheMood = {}
count = 0.0
tag_clear_previous = "zzz@@@zzz"
for row in depecheMood:
   items = row.split("\t")
   vector = map(lambda x: float(re.sub("[^0-9.-]", "", x)), items[1:])
   tag = items[0]
   tag_clear = tag[:tag.rfind("#")]
   if tag_clear_previous != tag_clear:
      if tag_clear_previous in dict_depecheMood:
         dict_depecheMood[tag_clear_previous] = map(lambda x: x / count, dict_depecheMood[tag_clear_previous]) 
      count = 0.0
   if tag_clear in dict_depecheMood:
      vector_current = dict_depecheMood[tag_clear]
      dict_depecheMood[tag_clear] = map(lambda x, y: x + y, vector, vector_current)
      count += 1.0
   else:
      dict_depecheMood[tag_clear] = vector
      count = 1.0
   tag_clear_previous = tag_clear


def func_depecheMood(sentence):
   words = sentence.strip().replace("  ", " ").split(" ")
   n = len(words)
   matches_found = 0.0
   res_vector = [0.1] * 8
   for word in words:
      if word in dict_depecheMood:
         res_vector = map(lambda x, y: x + y, dict_depecheMood[word], res_vector)
         matches_found += 1.0 
   if matches_found == 0.0:
      to_ret = res_vector
   else:
      to_ret = map(lambda x: x / matches_found, res_vector)
   #print to_ret
   return to_ret

###############################################################################################



with open(os.path.join(cwd, 'NRC-Sentiment-Emotion-Lexicons', 'AutomaticallyGeneratedLexicons', 'NRC-Emoticon-AffLexNegLex-v1.0', 'Emoticon-AFFLEX-NEGLEX-unigrams.txt')) as data_file:
    rnc_Emoticon_AFFLEX_NEGLEX_unigrams = data_file.readlines()

dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams = {}

for row in rnc_Emoticon_AFFLEX_NEGLEX_unigrams:
   row = row.split("\t")[0:2]
   dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams[row[0]] = float(row[1])

def func_rnc_Emoticon_AFFLEX_NEGLEX_unigrams(sentence):
   #minn = 0.0
   avg = 0.0
   #maxx = 0.0
   #less_than_avg_cnt = 0.0
   #more_than_avg_cnt = 0.0
   NN = 2
   list_of_NN_min = []
   list_of_NN_max = []   
   sentence_tmp = sentence.strip().replace("  ", " ").split(" ")
   n = len(sentence_tmp)
   for i in range(n):
     for x in range(2):
       if x == 0: w = sentence_tmp[i]
       elif i < n-1: w = sentence_tmp[i] + " " + sentence_tmp[i+1]
       else: continue
       if w in dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams:
          val = dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams[w]
          #minn = min(minn, val)
          if len(list_of_NN_min) < NN or (len(list_of_NN_min) >= NN and val < list_of_NN_min[NN-1]):
             list_of_NN_min.append(val)
             list_of_NN_min.sort()
             if (len(list_of_NN_min) > NN):
                 list_of_NN_min = list_of_NN_min[:-1]
          elif len(list_of_NN_max) < NN or (len(list_of_NN_max) >= NN and val > list_of_NN_max[0]):
             list_of_NN_max.append(val)
             list_of_NN_max.sort()
             if (len(list_of_NN_max) > NN):
                 list_of_NN_max = list_of_NN_max[1:]
          avg += val
          
          #maxx = max(maxx, val)
   '''
   for i in range(n):
     for x in range(2):
       if x == 0: w = sentence_tmp[i]
       elif i < n-1: w = sentence_tmp[i] + " " + sentence_tmp[i+1]
       else: continue
       if w in dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams:
          val = dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams[w]
          if val < avg:
            less_than_avg_cnt += 1.0
          else:
            more_than_avg_cnt += 1.0
   '''
   #return [minn, less_than_avg_cnt/(float(n)), avg, more_than_avg_cnt/(float(n)), maxx]
   n_lmin = len(list_of_NN_min)
   n_lmax = len(list_of_NN_max)
   how_much_insufficient_min = NN - n_lmin
   how_much_insufficient_max = NN - n_lmax
   a = 0.0
   if n_lmin > 0:
       a = list_of_NN_min[n_lmin-1]
   for i in range(how_much_insufficient_min):
       list_of_NN_min.append(a)
   if n_lmax > 0:
       a = list_of_NN_max[n_lmax-1]
   for i in range(how_much_insufficient_max):
       list_of_NN_max.append(a) 
   list_of_NN_min.sort()
   list_of_NN_max.sort() 
   return [avg] + list_of_NN_min + list_of_NN_max


##########################################################################################


class word_embeddings_class:

   def __init__(self, dimensions_number, name):
         self.default_vector = (np.array([0.0] * dimensions_number, dtype=np.float16)).astype(np.float16)
         self.word_emb_dict = {}
         self.name = name


   def load(self):
      if self.name == "biu":
         f = open(os.path.join(cwd, 'word_embeddings', 'biu', "lexsub_words"))
      elif self.name == "google":
         f = open(os.path.join(cwd, 'word_embeddings', 'google', "GoogleNews-vectors-negative300.txt"))
      elif self.name == "glove_twitter":
         f = open(os.path.join(cwd, 'word_embeddings', 'glove', "glove.twitter.27B", "glove.twitter.27B.200d.txt"))
      else:
         f = open(os.path.join(cwd, 'word_embeddings', 'glove', "glove.840B.300d.txt"))
      lines = f.readlines()
      if self.name == "biu" or self.name == "google":
         lines = lines[1:] 
      for row in lines:
         tmp = row.split(" ")
         if self.name == "biu":
            self.word_emb_dict[tmp[0]] = (np.array(tmp[1:-1], dtype = np.float16)).astype(np.float16)
         else:
            self.word_emb_dict[tmp[0]] = (np.array(tmp[1:], dtype = np.float16)).astype(np.float16)


   def are_lists_equal(self, a, b):
      if len(a) != len(b):
         return False
      for i in range(len(a)):
         if a[i] != b[i]:
            return False
      return True


   def get_word_emb(self, sentence):
      res_n_hashed = np.float16(0.0)
      res_n_nonhashed = np.float16(0.0)
      res_vector = self.default_vector
      maximum_likelyhood_estimate_vector = (self.default_vector).astype(np.float16)
      sentence_by_word = sentence.split(" ")
      hashed_flag = False
      for word in sentence_by_word:
         if self.is_hashed_word(word):
            hashed_flag = True
            res_vector = (np.add(res_vector, self.get_word_vector(word), dtype=np.float16)).astype(np.float16)
            if not self.are_lists_equal(res_vector, self.default_vector):
               res_n_hashed += 1.0
         if self.is_hashed_word(word):
            word = word[1:]
         maximum_likelyhood_estimate_vector = (np.add(maximum_likelyhood_estimate_vector, self.get_word_vector(self.clear_the_word(word)), dtype=np.float16)).astype(np.float16)
         if not self.are_lists_equal(maximum_likelyhood_estimate_vector, self.default_vector):
            res_n_nonhashed += 1.0
      tmp = (np.divide(maximum_likelyhood_estimate_vector, (res_n_nonhashed+1.0), dtype=np.float16)).astype(np.float16)
      if hashed_flag:
         return [res_n_hashed, (np.divide(res_vector, (res_n_hashed+1.0), dtype=np.float16)).astype(np.float16), tmp]
      else:
         return [res_n_hashed, tmp, tmp]


   def get_word_emb_min_max(self, sentence, min_or_max):
      res_n_hashed = 1.0
      res_n_nonhashed = 1.0
      res_vector = self.default_vector
      maximum_likelyhood_estimate_vector = self.default_vector
      sentence_by_word = sentence.split(" ")
      hashed_flag = False
      for word in sentence_by_word:
         if self.is_hashed_word(word):
            hashed_flag = True
            res_vector = self.min_or_max_func(res_vector, self.get_word_vector(word), min_or_max)
            if not self.are_lists_equal(res_vector, self.default_vector):
               res_n_hashed += 1.0
         elif not hashed_flag:
            maximum_likelyhood_estimate_vector = self.min_or_max_func(maximum_likelyhood_estimate_vector, self.get_word_vector(self.clear_the_word(word)), min_or_max)
            if not self.are_lists_equal(maximum_likelyhood_estimate_vector, self.default_vector):
               res_n_nonhashed += 1.0
      if hashed_flag:
         return res_vector
      else:
         return maximum_likelyhood_estimate_vector


   def min_or_max_func(self, res_vector, current_word_vector, min_or_max):
      n = len(res_vector)
      aggregated = 0
      for i in range(n):
         aggregated = current_word_vector[i] - res_vector[i]
      if min_or_max == 'min':
         if aggregated < 0:
            return current_word_vector
         else:
            return res_vector
      else:
         if aggregated > 0:
            return current_word_vector
         else:
            return res_vector
      return res_vector


   def clear_the_word(self, word):
      letters = list(word)
      return "".join(filter(lambda x: (x>="a" and x<="z") or x == "-", letters))

   
   def is_hashed_word(self, word):
      return word[0:1] == "#"
   

   def get_word_vector(self, word):
      word = self.clear_the_word(word)
      if word in self.word_emb_dict:
         return (np.array(self.word_emb_dict[word], dtype=np.float16)).astype(np.float16)
      return (self.default_vector).astype(np.float16)


print "loading word embedding vectors", time.strftime('%X %x %Z')
'''
biu_word_embeddings = word_embeddings_class(600, "biu")
biu_word_embeddings.load()
'''


print 2
glove_twitter_word_embeddings = word_embeddings_class(200, "glove_twitter")
glove_twitter_word_embeddings.load()

print 3
glove_common_crawl_word_embeddings = word_embeddings_class(300, "glove_common_crawl")
glove_common_crawl_word_embeddings.load()

print 4
google_word_embeddings = word_embeddings_class(300, "google")
google_word_embeddings.load()

print "finished word embedding vectors", time.strftime('%X %x %Z')

##########################################################################################

def cross_validation(number_of_validations, train_set, classes_dataset):
  acc_r = 0.0
  acc_s = 0.0
  train_size = len(train_set) / number_of_validations
  for i in range(number_of_validations):
    test_range = range(i*train_size, (i+1)*train_size)
    train_range = list(set(range(len(train_set))).difference(test_range))
    #print "1: ", test_range
    #print "2: ", train_range

    test_subset_data = list(np.array(train_set)[test_range])
    test_subset_classes = list(np.array(classes_dataset)[test_range])
    train_subset_data = list(np.array(train_set)[train_range])
    train_subset_classes = list(np.array(classes_dataset)[train_range])

    clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=3)
    clf.fit(train_subset_data, train_subset_classes)
    p = clf.predict(test_subset_data)

    r_row, p_value = scipy.stats.pearsonr(p, test_subset_classes)
    s_row, p_value = scipy.stats.spearmanr(p, test_subset_classes)
    print len(test_subset_data), len(train_subset_data)
    print r_row, s_row
    acc_r += r_row
    acc_s += s_row
  return acc_r / float(number_of_validations), acc_s / float(number_of_validations)



##########################################################################################

def prediction(train_x, train_y, test_x):
    clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=3)
    clf.fit(train_x, train_y)
    p = clf.predict(test_x)
    return p



##########################################################################################




result_test_set = []
print '-' * 50
cnt = 0
dictTextProcessingCom = { "neg": 0, "neutral": 1, "pos": 2 }
for affect in affect_list:
  train_x = []
  train_y = []
  test_x = []
  for train_part in [train_data]:
    #is_train_data = train_part == train_data
    #print "bool: ", is_train_data
    for idd in train_part[affect]:
        train_tweet_json = train_part[affect][idd]
        sentence = train_tweet_json["atext_clear_lower"]
        #vec = func_rnc_Emoticon_AFFLEX_NEGLEX_unigrams(sentence)
        #vec = [vec[0], vec[2], vec[4]]
        #vec = [vec[2]] 
        #tmp = vec
        #print tmp
        #if (not is_train_data) or (is_train_data and train_tweet_json["remained"] != 0):
        #try:
        #cnt += 1
        #if cnt % 500 == 0: print cnt
        #tmp0 = biu_word_embeddings.get_word_emb(train_data[i]['list'][x]["atext_clear_lower"])
        
        tmp1 = glove_twitter_word_embeddings.get_word_emb(sentence)
        tmp2 = glove_common_crawl_word_embeddings.get_word_emb(sentence)
        tmp3 = google_word_embeddings.get_word_emb(sentence)
        
        ''' 
        tmp2_min = biu_word_embeddings.get_word_emb_min_max(data[i]['list'][x]["atext_clear_lower"], 'min')
        tmp2_max = biu_word_embeddings.get_word_emb_min_max(data[i]['list'][x]["atext_clear_lower"], 'max')
        tmp3_min = glove_word_embeddings.get_word_emb_min_max(data[i]['list'][x]["atext_clear_lower"], 'min')
        tmp3_max = glove_word_embeddings.get_word_emb_min_max(data[i]['list'][x]["atext_clear_lower"], 'max')
        '''
        #print tmp0[0]
        #print tmp0[1]
        #print len(tmp0[1])
        
        tmp = (
          np.ma.concatenate([
             #vec,
			 func_depecheMood(sentence),
             analyzer.polarity_scores(sentence),
             np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
             np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
             np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16),
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp1[1],
             tmp1[2],
             tmp2[1],
             tmp2[2],
             tmp3[1],
             tmp3[2]
           ])
        )
        
        #print tmp
        #sys.exit(0)
        train_x.append(tmp)
        train_y.append(float(train_tweet_json['magnitude']))
      #except:
      #  print train_data[i]['list'][x]["id"]
      #  #print tmp1, tmp2, tmp3
      #  sys.exit(0)
 
    
  for idd in test_data[affect]:
        tweet_json = test_data[affect][idd]
        sentence = tweet_json["atext_clear_lower"]
        #vec = func_rnc_Emoticon_AFFLEX_NEGLEX_unigrams(sentence)
        #vec = [vec[0], vec[2], vec[4]]
        #vec = [vec[2]] 
        #tmp = vec
        #print tmp
        
        tmp1 = glove_twitter_word_embeddings.get_word_emb(sentence)
        tmp2 = glove_common_crawl_word_embeddings.get_word_emb(sentence)
        tmp3 = google_word_embeddings.get_word_emb(sentence)
        
        
        tmp = (
          np.ma.concatenate([
             #vec,
              func_depecheMood(sentence),
              analyzer.polarity_scores(sentence),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16),
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp1[1],
             tmp1[2],
             tmp2[1],
             tmp2[2],
             tmp3[1],
             tmp3[2]
           ])
        )
          
        test_x.append(tmp)

  '''
    train_size = len(data[i]['list']) / 2 * 1
    clf.fit(train[:train_size], classes_dataset[:train_size])
    p = clf.predict(train[train_size:])
    print data[i]['affect']
    r_row, p_value = scipy.stats.pearsonr(p, classes_dataset[train_size:])
    s_row, p_value = scipy.stats.spearmanr(p, classes_dataset[train_size:])
  '''
  #k_fold = 3
  #r_row, s_row = cross_validation(k_fold, train, classes_dataset)
  #print('Pearson correlation coefficient: %.4f' % (r_row))
  #print('Spearman correlation coefficient: %.4f' % (s_row))
    
  p = prediction(train_x, train_y, test_x)
    
  text_for_prediction_file = 'ID	Tweet	Affect Dimension	Intensity Score\n'
  iiii = 0
  for idd in test_data[affect]:
        same_text = (idd + '\t' + test_data[affect][idd]['atext'] + '\t' + affect + '\t')
        text_for_prediction_file += (same_text + str(p[iiii]) + '\n')
        iiii += 1
  f_prediction = open(os.path.join(cwd, 'dataset', 'task3', 'submission', 'V-reg_en_pred.txt'), 'w+')
  f_prediction.write(text_for_prediction_file)
  f_prediction.close()

    
  #print 'submission file for affect "%s" is made' % (data[i]['affect'])

  print '-' * 50

  #result_test_set = []
