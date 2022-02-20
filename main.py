import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
#create a title
st.title("streamlit example")
#write text with markdown language # eq to h1
st.write("""
# explore different classifiers 
# Whitch one is the best ?
""")
#create a select box first argument label second list of options and we can assigne the select box 
# to a var to get the value
#-------------------------------------code--------------------------------------------------
#dataset_name=st.selectbox("Select Dataset",("iris","Breast Cancer","Wine dataset"))
#st.write(dataset_name)
#-------------------------------------------------------------------------------------------------
#when we assigne a dataset the script is runing again and update the layout streamlit uses an intelligent way 
#to cache the scriot tht do not need to run

# we can create a sidebar  selector
dataset_name=st.sidebar.selectbox("Select Dataset",("iris","Breast Cancer","Wine dataset"))
classifier_name=st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random forest"))
def get_dataset(dataset_name):
    if dataset_name=="iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y

X,y=get_dataset(dataset_name)
st.write("shape of dataset",X.shape)
st.write("number of classes",len(np.unique(y)))
        

def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        #we are adding ui  for the K knn  classifier parameter we specify th elements to add in params cz we gonna give the ability to manage for eatch cli 
        #the slider put in k has 3 parameters label start end
        k=st.sidebar.slider("K",1,15)
        params["K"]=k
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params=add_parameter_ui(classifier_name)

#classification with sklearn
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
#reduction de la dim pr pouvoir afficher sur 2 dim

pca = PCA(2)
#trans red avc pca
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
#par 1 axis des cord p2 axis des x 2 des alpha Transparence and cmap for cle des couleurs
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
