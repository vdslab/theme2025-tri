from flask import Flask, request
from flask_cors import CORS
from main import get_data

app = Flask(__name__)
CORS(app)

@app.get("/api/clusterVisualization/logistic-regression-data")
def get_logistic_regression_data():
    gamepk = request.args.get('gamepk', type=int)
    
    if gamepk is None:
        return {"error": "gamepk parameter is missing"}
    
    try:
        data = get_data(gamepk)
        if data is None:
            return {"error": "data is None"}
        return data
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)