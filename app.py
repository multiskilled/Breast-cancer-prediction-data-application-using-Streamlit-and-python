import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing



st.title("Breast Cancer Prediction")


st.markdown("""About App: This app performs Breast Cancer Prediction.""")

#file upload
file_bytes=st.file_uploader("Upload a File", type="csv")

#check if the file upload is sucessful or not.
if file_bytes is not None:
    def load_data(path):
        data=pd.read_csv(path) #path of the uploaded csv file
        data=data.drop("id",axis=1) #delete the id column
        data=data.fillna(data.mode())
        return data
#Transform non numeric data with label encoder.
    cpd=load_data(file_bytes)

##clean the data
    def cleaning(data):
        label_en=preprocessing.LabelEncoder()
        data1=data
        for i in data1.columns:
            cols=data1[i].dtypes
            if cols=='object': #converting the type of column
                data1[i]=data1[[i]].astype(str).apply(label_en.fit_transform) #subsetting the column and apply LE
            else:
                data1[i]=data1[i] #no changes
        return data1
    cleaned_data=cleaning(cpd)

#select the dependent and independent variables
    st.sidebar.header('select Output Variable')
    columns_names=list(cpd.columns)
    Dependent_var=st.sidebar.selectbox('Dependent Variables',columns_names)
    #after loadin the data columns, you can remove them by selecting x
    columns_names.remove(Dependent_var)
    st.sidebar.header("Unselect variables which you think are not important for analysis")
    independent_var=st.sidebar.multiselect("Independent Variables",columns_names,columns_names)



    #split test vs train
    X=cleaned_data[independent_var] #define X features
    y=cleaned_data[Dependent_var] #define target variable
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4,random_state=0)


    #model
    classifier=LogisticRegression()
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    y_predd=["Not Affected" if i==1 else "Affected" for i in y_pred]

    #measures
    #confusion matrix
    con_matrix=confusion_matrix(y_test,y_pred)

    #predict probabilities
    Lr_probs=classifier.predict_proba(X_test)
    #keep probabilities for the positive outcome only
    Lr_probs=Lr_probs[:,1]

    #precision and recall
    lr_precision,lr_recall,_=precision_recall_curve(y_test,Lr_probs)

    #f1 and AUC
    lr_f1,lr_auc=f1_score(y_test,y_pred),auc(lr_recall,lr_precision)


    #display the measures
    X_test['Prediction']=y_predd
    X_test['Actual']=y_test

    st.write('Actual Data Dimension:'+ str(cpd.shape[0])+'rows and '+ str(cpd.shape[1])+' columns.')
    st.dataframe(cpd)
    st.write('Test Data Dimension:'+ str(X_test.shape[0])+'rows and '+ str(X_test.shape[1])+' columns.')
    st.dataframe(X_test)
    st.write("Confusion Matrix")
    st.write(con_matrix)
    st.write('Accuracy:'+str(accuracy_score(y_test,y_pred)))
    st.write('LogisticRegression : f1=%.3f auc=%.3f'%(lr_f1,lr_auc))