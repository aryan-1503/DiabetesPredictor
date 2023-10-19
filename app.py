from flask import Flask, render_template, request
import numpy as np
import pickle

filename = 'diabetes-prediction-model.pkl'
clf = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result.html',methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        gluc = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        ins = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg,gluc,bp,st,ins,bmi,dpf,age]])
        my_prediction = clf.predict(data)

        return render_template('result.html ',prediction=my_prediction)
        


if __name__ == '__main__':
    app.run(debug=True)