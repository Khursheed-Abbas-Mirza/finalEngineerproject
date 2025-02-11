# Description: This file contains the code for the API that will be used to predict the obesity level of a person.
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
app=Flask("Obseity Model")
CORS(app)
ob=joblib.load('obseity.pkl')
sc=joblib.load('scaler.pkl')

def predict_obesity(features):
    features = np.array(features).reshape(1, -1)
    features = sc.transform(features)
    prediction = ob.predict(features)
    return prediction[0]
@app.route('/api/check',methods=["POST"])
def check():
    data=request.get_json()
    bmi=round(float(data['weight'])/(float(data['height'])/100)**2)
    gender=1 if data['gender']=='Male' else 2
    cdata=[gender,int(data['Age']),data['height'],data['weight'],bmi]
    predict=predict_obesity(cdata)
 
    return jsonify({"succuess":True,"obeseity_level":predict})
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True,host="localhost",port=5000)