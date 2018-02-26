# -*- coding: utf-8 -*-
# encoding=utf8
import sys, json, re
reload(sys)
sys.setdefaultencoding('utf8')
import os
cwd = os.getcwd()
print cwd
tmp = []
type_list = ["dev", "test", "train"]
for typee in type_list:
  lines = []
  with open(os.path.join(cwd, "raw", "2018-E-c-En-" + typee + ".txt")) as data_file:
    tmp = data_file.readlines()
  lines = tmp[1:]
  print len(lines)
  data_res = {}
  for row in lines:
    #print row
    tmp = row.split("\t")
    data_res[tmp[0]] = {}
    print tmp[0]
    data_res[tmp[0]]["atext"] = tmp[1]
    sss = re.sub("[^a-zA-Z*'_#@0-9]", " ", tmp[1])
    data_res[tmp[0]]["atext_clear_lower"] = sss.lower()
   
    classes_list = []
    for i in range(11):
      if tmp[i+2].replace("\n", "").replace("\r", "") == "1":
         classes_list.append(i)
    data_res[tmp[0]]["classes"] = classes_list


  import json
  a = json.dumps(data_res, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
  f = open(os.path.join(cwd, typee, "dataset_" + typee + "_2.txt"), 'w+')
  f.write(a)
  f.close()
