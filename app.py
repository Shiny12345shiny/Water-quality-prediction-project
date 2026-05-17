from flask import Flask, render_template, request
import numpy as np
import joblib as jb
from pathlib import Path


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ph=float(request.form['ph'])
    hardness=float(request.form['hardness'])
    solids=float(request.form['solids'])
    chloramines=float(request.form['chloramines'])
    sulfate=float(request.form['sulfate'])
    conductivity=float(request.form['conductivity'])
    organic_carbon=float(request.form['organic_carbon'])
    trihalomethanes=float(request.form['trihalomethanes'])
    turbidity=float(request.form['turbidity'])

    arr=np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    # Load the model
    model=jb.load(Path("model/model_rfc.pkl"))
    
    # Make a prediction
    prediction=model.predict(arr)
    if prediction[0]==1:
        return render_template('result.html', result='The water is safe to drink.')
    else:
        return render_template('result.html', result='The water is not safe to drink.')


if __name__ == '__main__':
    app.run(debug=True)
