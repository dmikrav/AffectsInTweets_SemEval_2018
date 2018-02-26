# -*- coding: utf-8 -*-
# encoding=utf8
import sys, json, re
reload(sys)
sys.setdefaultencoding('utf8')
import os
cwd = os.getcwd()
print cwd
lines = []
tmp = []
data_res = {}
preffix_list = ["anger", "fear", "joy", "sadness"]
suffix = "-test.txt"
path_str = os.path.join(cwd, "raw", "2018-EI-reg-En-test")
#lst_file_names = os.listdir(path_str)
for affect in preffix_list:
  data_res[affect] = {}
  with open(os.path.join(path_str, affect + suffix)) as data_file:
    tmp = data_file.readlines()
  lines = tmp[1:]
  for row in lines:
    tmp = row.split("\t")
    #print res_dict["id"]
    res_dict = {}
    res_dict["atext"] = tmp[1]
    sss = re.sub("[^a-zA-Z*'_#@0-9]", " ", tmp[1])  
    res_dict["atext_clear_lower"] = sss.lower()
    #res_dict["magnitude"] = float(tmp[3])
    data_res[affect][tmp[0]] = res_dict

a = json.dumps(data_res, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
f = open(os.path.join(cwd, 'test', "dataset_json_test.txt"), 'w+')
f.write(a)
f.close()
