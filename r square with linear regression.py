import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("linear_regression.csv",sep=";")

#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% Linear Regression
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)
linear_reg.fit(x,y)
y_head=linear_reg.predict(x)

plt.plot(x,y_head,color="red")


#%%
from sklearn.metrics import r2_score
print("r_square source: ",r2_score(y,y_head))






