from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os

from src.gemstonePricePrediction.pipeline.prediction import CustomData, PredictionPipeline

app = Flask(__name__)

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")
    else:
        try:
            data = CustomData(
                carat=float(request.form.get("carat")),
                depth=float(request.form.get("depth")),
                table=float(request.form.get("table")),
                x=float(request.form.get("x")),
                y=float(request.form.get("y")),
                z=float(request.form.get("z")),
                cut=request.form.get("cut"),
                color=request.form.get("color"),
                clarity=request.form.get("clarity")
            )
            
            pred_df = data.get_data_as_dataframe()
            print(pred_df)
            
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template("index.html", results=round(results[0], 2))
            
        except Exception as e:
            return render_template("index.html", results="Error occurred")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
