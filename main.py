import numpy as np 
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
import pickle as pickle
from ast import literal_eval

rank = np.genfromtxt('test.out', delimiter=',', dtype=None, names=('model', 'accuracy', 'auc_score', 'fscore', 'recall', 'precision', 'fpr'))

ret = []
for i in range(len(rank)):
    app = []
    for j in range(len(rank[0])):
        app.append(rank[i][j])
    ret.append(app)


modelList = pd.DataFrame.from_records(ret)
cols = ['model', 'accuracy', 'auc_score', 'fscore', 'recall', 'precision', 'fpr']
modelList.columns = cols
fpr_first = modelList.sort_values(by=['fpr', 'accuracy', 'auc_score'], ascending=[True, False, False])



app = Flask(__name__)

@app.route('/')
def hello_world():
    return """<h1>Hello</h1>
            <h2>We use a HYBRID OF SMOTE AND RANDOM UNDERSAMPLING TO DEAL WITH IMBALANCE</h2>
            <p>Go to <a href="/home">home</a> to enter data</p>
            <img src="static//smoterus_report.png" alt="Confusion Matrix"> """

@app.route('/home', methods=['POST', 'GET'])
def getvals():
    if request.method == 'POST':
        ret = []
        time = request.form['Time']
        ret.append(time)
        for i in range(1, 29):
            ret.append(request.form[str('V') + str(i)])
        amt = request.form['Amount']
        ret.append(amt)
        return redirect(url_for('prediction', vels = ret))
    else:
        return render_template('vis.html')
    

@app.route('/prediction/<vels>')
def prediction(vels):
    vels = literal_eval(vels)
    inp = []
    for i in vels:
        inp.append(float(i))
    inp = np.array(inp)
    pipe = pickle.load(open('SMOTERUS.pkl', 'rb'))
    out = pipe.predict([inp])
    if out == [1]:
        return """<h1>Fraud</h1>"""
    else:
        return """<h1>Not Fraud</h1>"""

if __name__ == '__main__':
    app.run(debug=True)
