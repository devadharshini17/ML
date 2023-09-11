import pandas as pd
import numpy as np
data=pd.read_csv("enjoysport.csv")
print(data)
d=np.array(data)[:,:-1]
print("the attributes are:",d)
target=np.array(data)[:,-1]
print("tagert :",target)
def train(c,t):
    for i,val in enumerate(t):
        if val=="yes":
            sh=c[i].copy()
            break
    for i,val in enumerate(c):
        if t[i]=="yes":
            for x in range(len(sh)):
                if val[x]!=sh[x]:
                    sh[x]='?'
                else:
                    pass
    return sh
print("the final sh:",train(d,target))
