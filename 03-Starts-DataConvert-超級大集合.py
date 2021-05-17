import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

#read data
df = pd.read_csv('heart.csv.xls',encoding='utf-8')
print(df.head())
############
# 文字分類 轉 數字分類
#df["ColorNo"]=df.Color.astype("category").cat.codes                     # 文字分類轉成數字
#df["Spectral_ClassNo"]=df.Spectral_Class.astype("category").cat.codes   # 文字分類轉成數字






#############
print("資料拆切---")
# 決定X 分類 和Y分類 要用的欄位
dfX=df[['age','sex','cp','trtbps','chol','fbs',
       'restecg','exng','oldpeak','slp','caa','thall']]
dfY=df['thall']
X=dfX.to_numpy()
Y=dfY.to_numpy()
X_train ,X_test ,Y_train ,Y_test = train_test_split(X,Y,test_size=0.1)

#############
print("迴歸計算----------------------------------------")
from sklearn import linear_model
reg = linear_model.LinearRegression()    # 初使化
reg.fit(X_train,Y_train)
Y_test_predict= reg.predict(X_test)
print("regr.coef_ 係數:",reg.coef_)
print("reg.singular_ 單數:",reg.singular_)
print("---")
print(".               實際答案:",Y_test)
print("LinearRegression預測答案:",Y_test_predict)

print("KNN計算----------------------------------------")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
Y_test_predict=knn.predict(X_test)
print(".               實際答案:",Y_test)
print("KNN             預測答案:",Y_test_predict)
print('                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率

print("K-means計算------------------------------------")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
kmeans.fit(X_train)
Y_test_predict=kmeans.predict(X_test)
print(".               實際答案:",Y_test)
print("K-means         預測答案:",Y_test_predict)
print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率
#print("第0個分類的位置",kmeans.cluster_centers_[:,0])
#print("第1個分類的位置",kmeans.cluster_centers_[:,1])

print("決策數計算--------------------------------------")
from sklearn import tree
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree.fit(X_train,Y_train)
Y_test_predict=DecisionTree.predict(X_test)
tree.export_graphviz(DecisionTree,out_file='tree.dot')

print(".               實際答案:",Y_test)
print("決策數           預測答案:",Y_test_predict)
print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率

print("隨機forest計算--------------------------------------")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,Y_train)
Y_test_predict=model.predict(X_test)

from sklearn.tree import export_graphviz
estimator = model.estimators_[5]
export_graphviz(estimator,out_file='forest.dot')

print(".               實際答案:",Y_test)
print("隨機forest       預測答案:",Y_test_predict)
print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率



print("NB       計算--------------------------------------")
from sklearn.naive_bayes import GaussianNB
model =GaussianNB()
model.fit(X_train,Y_train)
Y_test_predict=model.predict(X_test)

print(model.class_prior_ )
print(model.get_params() )
print(".               實際答案:",Y_test)
print("NB              預測答案:",Y_test_predict)
print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率



print("Lass      計算--------------------------------------")
from sklearn import linear_model


model =linear_model.Lasso(alpha=0.1)
model.fit(X_train,Y_train)
Y_test_predict=model.predict(X_test)


print(".               實際答案:",Y_test)
print("Lass            預測答案:",Y_test_predict)
# print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict))    #算準確率

print("SGDClassifier      計算--------------------------------------")
from sklearn.linear_model import SGDClassifier
model =SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
model.fit(X_train,Y_train)
Y_test_predict=model.predict(X_test)

print(".               實際答案:",Y_test)
print("SGDClassifier   預測答案:",Y_test_predict)
print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率

print("GaussianProcessRegressor      計算--------------------------------------")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


kernel = DotProduct() + WhiteKernel()
model= GaussianProcessRegressor(kernel=kernel, random_state=0)

model.fit(X_train,Y_train)
Y_test_predict=model.predict(X_test)

print(".               實際答案:",Y_test)
print("GaussianProcessRegressor   預測答案:",Y_test_predict)
# print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率


print("svm.SVR       計算--------------------------------------")
from sklearn import svm

model= svm.SVR()
model.fit(X_train,Y_train)
Y_test_predict=model.predict(X_test)

print(".               實際答案:",Y_test)
print("svm.SVR   預測答案:",Y_test_predict)
# print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率


print("svm.SVC      計算--------------------------------------")
from sklearn.model_selection import cross_val_score
model= svm.SVC(kernel='linear', C=1, random_state=42)
model.fit(X_train,Y_train)
Y_test_predict=model.predict(X_test)

print(".               實際答案:",Y_test)
print("svm.SVC         預測答案:",Y_test_predict)
# print('.                 準確率:',metrics.accuracy_score(Y_test,Y_test_predict) )    #算準確率



#############
print("畫seaborn 圖---")
df2=df[['age','sex','cp','trtbps','chol','fbs',
       'restecg','exng','oldpeak','slp','caa','thall']]
sns.set_theme(style="ticks")
#sns.set_theme(style="whitegrid")
sns.pairplot(df2, hue="thall")
plt.show()
