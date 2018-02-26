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
file_list = ["dev", "train", "test"]
affect = "valence"
for file_name in file_list:
    path_str = os.path.join(cwd, "raw", '2018-Valence-reg-En-' + file_name + '.txt')
    with open(os.path.join(path_str)) as data_file:
      tmp = data_file.readlines()
    lines = tmp[1:]
    data_res[affect] = {}
    for row in lines:
      tmp = row.split("\t")
      res_dict = {}
      res_dict["atext"] = tmp[1]
      sss = re.sub("[^a-zA-Z*'_#@0-9]", " ", tmp[1])  
      res_dict["atext_clear_lower"] = sss.lower()
      if file_name != "test":
          res_dict["magnitude"] = float(tmp[3])
      data_res[affect][tmp[0]] = res_dict

    a = json.dumps(data_res, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
    f = open(os.path.join(cwd, file_name, "dataset_json_" + file_name + ".txt"), 'w+')
    f.write(a)
    f.close()
