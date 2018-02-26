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
suffix = "-train.txt"
path_str = os.path.join(cwd, "raw", "EI-reg-English-Train")
f = open(os.path.join(cwd, 'train', "dataset_json_task_1.txt"))
train_data_old = json.load(f)
f.close()
f = open(os.path.join(cwd, 'train', "new_magnitudes.txt"))
train_data_new = json.load(f)
f.close()
#lst_file_names = os.listdir(path_str)
for affect in preffix_list:
  for idd in train_data_old[affect]:
     try:
        train_data_old[affect][idd]["remained"] = 1
        train_data_old[affect][idd]["magnitude"] = train_data_new[affect]["2017-En-"+idd]["magnitude"]
     except:
        train_data_old[affect][idd]["remained"] = 0
  
  for idd in train_data_new[affect]:
     tmp = idd.split("-")
     idd_old_formated = str(int(tmp[2]))
     if not idd_old_formated in train_data_old[affect]:
        train_data_old[affect][idd_old_formated] = train_data_new[affect][idd]
        train_data_old[affect][idd_old_formated]["remained"] = 2
  
a = json.dumps(train_data_old, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
f = open(os.path.join(cwd, 'train', "dataset_json_task_1_new_magnitute.txt"), "w+")
f.write(a)
f.close()
