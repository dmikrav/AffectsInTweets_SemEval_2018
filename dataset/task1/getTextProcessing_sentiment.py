	
# coding: utf-8

# In[41]:

#import urllib2, json
from subprocess import Popen, PIPE

def get_json(text):
    verbose = False
    finished = True
    jsonn = {}
    for i in range(3):
        try:
            process = Popen('curl -d "text=' + text + '" http://text-processing.com/api/sentiment/', stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            return stdout
        except:
            pass
    return jsonn


# In[40]:

#print get_json('check')


# In[22]:



import sys, json, re
reload(sys)  
sys.setdefaultencoding('utf8')
import os
cwd = os.getcwd()
print cwd


# In[35]:

with open(os.path.join(cwd, 'test', "dataset_json_test.txt")) as data_file:    
    data = json.load(data_file)


# In[43]:
i = 0
excepted = 0
done = 0
preffix_list = ["anger", "fear", "joy", "sadness"]
for aff in preffix_list:
  if excepted > 17:
      break
  for idd in data[aff]:
        if excepted > 17:
            break
        ##if link['metrics'][0]['textProcessingCom']['label'][0:3] == 'BAD':
        #if data[aff][idd]["remained"] == 2:
        if not ('metrics' in data[aff][idd] and 'textProcessingCom' in data[aff][idd]['metrics']):
          try:
            data[aff][idd]['metrics'] = {}
            data[aff][idd]['metrics']['textProcessingCom'] = json.loads(get_json(data[aff][idd]['atext_clear_lower']))
            done += 1
            excepted = 0
          except:
            excepted += 1
        i += 1
        print i, done, excepted
            #get_json(item['text'])


# In[1]:




# In[1]:




# In[44]:

import json
a = json.dumps(data, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
f = open(os.path.join(cwd, 'test', "dataset_json_test_2.txt"), 'w+')
f.write(a)
f.close()


# In[ ]:



