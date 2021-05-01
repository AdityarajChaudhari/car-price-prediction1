import numpy as np
import pickle
from flask import Flask,render_template,request,jsonify
from flask_cors import CORS,cross_origin


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@cross_origin()
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
@cross_origin()
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('index.html',prediction_text ="Employee salary should be $ {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)