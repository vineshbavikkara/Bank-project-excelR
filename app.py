#  For Deployment
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from xgboost import XGBClassifier
import pickle


app  = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    response=[x for x in request.form.values()]  
    
    response = pd.DataFrame(response).T
    response.columns = ['DisbursementGross', 'Term', 'NoEmp', 'NewExist_new','FranchiseCode_yes',
                    'UrbanRural_urban','LowDoc_yes', 'RevLineCr_yes']

    # def of changing to dummy variables value
    def dummy_urban(num):
        if num is 'Urban':
            return 1
        
        else:
            return 0
    
    UrbanRural_urban = response['UrbanRural_urban'].apply(dummy_urban)

    # def of changing to dummy variables value
    def dummy_undefined(num):
        if num is 'Undefined':
            return 1
        
        else:
            return 0
        
    UrbanRural_undefined = response['UrbanRural_urban'].apply(dummy_undefined)
    
    dummy_data = pd.concat([UrbanRural_urban,UrbanRural_undefined], axis= 1, ignore_index= False)
    dummy_data.columns = ['UrbanRural_urban','UrbanRural_undefined']
    
    Cut_first = response.iloc[:,0:5]
    Cut_last = response.iloc[:,6:]
    
    response = pd.concat([Cut_first,dummy_data,Cut_last],axis=1,ignore_index=False)

    response = response.astype(int)
    pred = model.predict(response)
    
    if pred[0]==1:
        return render_template("index.html",predicted="High Risk")
    return render_template("index.html",predicted="Low Risk")



if __name__=="__main__":
    app.run(debug=True)
    