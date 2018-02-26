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
suffix = "-dev.txt"
path_str = os.path.join(cwd, "raw", "2018-EI-reg-En-dev")
f = open(os.path.join(cwd, 'development', "dataset_json_development.txt"))
dev_data_old = json.load(f)
f.close()
#lst_file_names = os.listdir(path_str)
for affect in preffix_list:
  f = open(os.path.join(path_str, affect+suffix))
  dev_data_new = f.readlines()
  f.close()
  dev_data_new = dev_data_new[1:]
  for row in dev_data_new:
     tmp = row.split("\t")
     idd = tmp[0]
     mag = float(tmp[3])
     dev_data_old[affect][idd]["magnitude"] = mag
 
  
a = json.dumps(dev_data_old, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
f = open(os.path.join(cwd, 'development', "dataset_json_development_new_magnitute.txt"), "w+")
f.write(a)
f.close()
