preffix_list = ["anger", "fear", "joy", "sadness"]
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
for affect_dict in train_data:
   affect = affect_dict["affect"]
   data_res[affect] = {}
   for link in affect_dict["list"]:
      data_res[affect][link["id"]] = link
      data_res[affect][link["id"]].pop('id', None)

a = json.dumps(data_res, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
f = open(os.path.join(cwd, 'train', "dataset_json_task_1_refactored.txt"), 'w+')
f.write(a)
f.close()
