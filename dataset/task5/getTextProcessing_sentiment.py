	
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
            cmd = 'curl -d "text=' + text + '" http://text-processing.com/api/sentiment/'
            print cmd
            process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            print 'after'
            stdout, stderr = process.communicate()
            print stdout
            return stdout
        except Exception as e:
            print "no result", i, e
    return jsonn


# In[40]:

d = get_json('check')
print d
if d == {}:
   print "exiting.."
   exit(1)


# In[22]:



import sys, json, re
reload(sys)  
sys.setdefaultencoding('utf8')
import os
cwd = os.getcwd()
print cwd


# In[35]:

type_list = ["dev", "test", "train"]

for typee in type_list:

  with open(os.path.join(cwd, typee, "dataset_" + typee + ".txt")) as data_file:
    data = json.load(data_file)


  # In[43]:
  cnt = 0
  cnt_exc = 0
  cnt_exc_total = 0
  succ = 0
  for idd in data:
    if cnt_exc > 47:
        break
    #print data[idd]['atext_clear_lower']
    cnt += 1
    if not 'metrics' in data[idd] or not 'textProcessingCom' in data[idd]['metrics']:
     try:
        #print data[idd]['atext_clear_lower']
        aaa = get_json(data[idd]['atext_clear_lower'])
        data[idd]['metrics'] = {}
        data[idd]['metrics']['textProcessingCom'] = json.loads(aaa)
        succ += 1
        cnt_exc = 0
     except:
        print
        cnt_exc += 1
        cnt_exc_total += 1
    print cnt, succ, cnt_exc, cnt_exc_total


            #get_json(item['text'])


# In[1]:




# In[1]:




# In[44]:

  import json
  a = json.dumps(data, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
  f = open(os.path.join(cwd, typee, "dataset_" + typee + "_3.txt"), 'w+')
  f.write(a)
  f.close()


# In[ ]:



