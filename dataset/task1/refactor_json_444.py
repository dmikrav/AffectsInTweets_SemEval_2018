affect_list = ["anger", "fear", "joy", "sadness"]
suffix = "-train.txt"
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
f = open(os.path.join(cwd, 'train', "dataset_json_task_1.txt"))
train_data = json.load(f)
f.close()
#lst_file_names = os.listdir(path_str)
for affect in affect_list:
   for idd in train_data[affect]:
      if isinstance(train_data[affect][idd]["metrics"], list):
         train_data[affect][idd]["metrics"] = train_data[affect][idd]["metrics"][0]

a = json.dumps(train_data, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
f = open(os.path.join(cwd, 'train', "dataset_json_task_1_upd.txt"), 'w+')
f.write(a)
f.close()
