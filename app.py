import numpy as np
import pandas as pd
from flask import Flask,render_template,request,jsonify
import pickle
import sklearn
from flask_cors import CORS,cross_origin

app = Flask(__name__)
model= pickle.load(open('model.pkl','rb'))
@cross_origin()
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@cross_origin()
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        Present_Price = float(request.form['Showroom Price'])
        Kms_Driven = int(request.form["No.of KM's Driven"])
        Kms_Driven2 = np.log(Kms_Driven)
        second_hand = request.form['Is_the_car_second_hand']
        if second_hand == 'No':
            second_hand = 0
        else:
            second_hand = 1
        seller_type = request.form['Type of Seller']
        if seller_type == 'Individual' :
            seller_type = 0
        else:
            seller_type = 1
        fuel_type = request.form['Type of Fuel']
        if fuel_type=='Petrol' :
            fuel_type = 1
        else:
            fuel_type = 0
        transmission = request.form['Transmission']
        if transmission == 'Manual':
            transmission = 0
        else:
            transmission = 1


        features = ([[Present_Price,Kms_Driven2,fuel_type,seller_type,transmission,second_hand]])
        features = np.array(features)
        predictions = model.predict(features)
        output = round(predictions[0],2)
        if output<0:
            return render_template('index.html',prediction_text = "You cannot sell the car!!")
        else:
            return render_template('index.html',prediction_text =f"You can sell this car at {output}price!!")
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)


