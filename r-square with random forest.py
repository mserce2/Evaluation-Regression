"""
Bu derste random forest algoritmasında ki
bulduğumuz değerler gereçeğe ne kadar yakın
test edip sonucu ekrana yazdıracağız
"""

import pandas as pd
import numpy as np
df=pd.read_csv("r-square.csv",sep=";",header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)
y_head=rf.predict(x)


#%%
#Bulduğumuz değerlerin gerçeğe ne kadar yakın
#olduğunu metrics kütüphanesi ile anlayabiliriz
from sklearn.metrics import r2_score
print("r_source: ",r2_score(y,y_head))
#sonuc 1 e ne kadar yakın olursa algoritma o kadar başaralı demektir
