# -*- coding: utf-8 -*-
# encoding=utf8
import sys, json, re
reload(sys)
sys.setdefaultencoding('utf8')
import os
cwd = os.getcwd()
print cwd
lines_template = [] 
lines_submission = []
dict_of_predicted = {}
tmp = []
if True:
    path_str = os.path.join(cwd, "submission", "V-reg_en_pred.txt")
    with open(os.path.join(path_str)) as data_file:
      tmp = data_file.readlines()
    lines_submission = tmp[1:]
    
    for row in lines_submission:
        ttt = row.split("\t")
        dict_of_predicted[ttt[0]] = ttt[3]

    path_str = os.path.join(cwd, "raw", "2018-Valence-reg-En-dev.txt")
    with open(os.path.join(path_str)) as data_file:
      tmp = data_file.readlines()
    headers = tmp[0]
    lines_template = tmp[1:]

    for row in lines_template:
        ttt = row.split("\t")
        ttt[3] = dict_of_predicted[ttt[0]]
        row = "\t".join(ttt)

    result = "".join([headers] + lines_template)
    
    f = open("V-reg_en_pred.txt", "w+")
    f.write(result)
    f.close()
