from flask import Flask, render_template, request
import pandas as pd
from anomaly_model import AnomalyDetector

app = Flask(__name__)
detector = AnomalyDetector()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        data = pd.read_csv(file)
        detector.load_data(file)
        detector.train_model()
        predictions = detector.predict(data)
        result = list(predictions)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
