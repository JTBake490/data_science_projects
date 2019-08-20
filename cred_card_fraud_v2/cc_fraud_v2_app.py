import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
with open('cc_fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

feature_names = ['Time',
 'V1',
 'V2',
 'V3',
 'V4',
 'V5',
 'V6',
 'V7',
 'V8',
 'V9',
 'V10',
 'V11',
 'V12',
 'V13',
 'V14',
 'V15',
 'V16',
 'V17',
 'V18',
 'V19',
 'V20',
 'V21',
 'V22',
 'V23',
 'V24',
 'V25',
 'V26',
 'V27',
 'V28',
 'Amount']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    x_input= [float(request.form.get(name)) for name in feature_names]
    print(x_input)
    final_features = [np.array(x_input)]
    prediction = model.predict_proba(final_features).reshape(1, -1)[:,1]>0.6001

    if prediction == False:
        prediction = 'Legitamate'
    else:
        prediction = 'Fraudulent'

    return render_template('index.html', prediction_text=f'{prediction} Transaction')

if __name__ == '__main__':
    app.run(debug=True)