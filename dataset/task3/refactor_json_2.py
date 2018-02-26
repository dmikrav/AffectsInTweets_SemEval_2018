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
f = open(os.path.join(cwd, 'test', "dataset_json.txt"))
test_data = json.load(f)
f.close()
#lst_file_names = os.listdir(path_str)
for affect in preffix_list:
   data_res[affect] = {}
   for link in test_data[affect]:
      data_res[affect][link["id"]] = link
      data_res[affect][link["id"]].pop('id', None)

a = json.dumps(data_res, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
f = open(os.path.join(cwd, 'test', "dataset_json.txt"), 'w+')
f.write(a)
f.close()
