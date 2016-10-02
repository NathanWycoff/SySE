import pandas as pd
import code
import subprocess
import commands
import re

df = pd.read_csv('data_sentences')

sentences = df['0']

sentences = [re.sub('\"','',x) for x in sentences]

output_list = []
i = 0
for x in sentences:
    i += 1
    print 'Sentence ' + str(i)
    #print 'echo \"' + x + '\" | syntaxnet/demo1.sh'
    output_list.append(commands.getstatusoutput('echo \"' + x + '\" | syntaxnet/demo1.sh'))



code.interact(local=locals())


