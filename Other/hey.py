from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
from flask_restful import reqparse
app = Flask(__name__)
@app.route("/", methods=['GET'])
def hello():
    return "hey"
if __name__ == '__main__':
    app.run(debug=True)