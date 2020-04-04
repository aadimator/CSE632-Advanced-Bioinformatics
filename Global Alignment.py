

import pandas as pd

from Bio import pairwise2
from Bio.SubsMat.MatrixInfo import blosum62
from Bio import SeqIO
from Bio.Seq import Seq

from Bio.SubsMat import SeqMat


df = pd.read_csv("inputs/score_matrix_small.csv", header=0, index_col=0)
df


# In[4]:


data = df.stack().to_dict()
data


# In[5]:


seq1 = Seq("FACEDCAFFE")
seq2 = Seq("ACEDFACEDFACED")


# In[6]:


submat = SeqMat(data)


# In[7]:


gop = -10; gep = -0.5
alignments = pairwise2.align.globalds(seq1, seq2, submat, gop, gep)
alignments


# In[8]:


print(pairwise2.format_alignment(*alignments[0]))


# In[9]:


x = list(alignments[0][0])
print(x)


# In[10]:


y = list(alignments[0][1])
y


# In[11]:


import matplotlib.pyplot as plt
import numpy as np


# In[12]:


names = np.array(list(x))
c = np.random.randint(1,5,size=15)
print("this is C")
print(c)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = plt.scatter(x,y, s=100)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()


# In[13]:


plt.plot(list(x), list(y), "ro")


# In[ ]:





# In[ ]:




