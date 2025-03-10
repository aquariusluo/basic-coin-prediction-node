import json
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

app = Flask(__name__)

def update_data():
    """Download price data, format data, and train the model."""
    try:
        if MODEL == "kNN" and DATA_PROVIDER == "binance":
            print("🚀 Updating Allora worker: Fetching fresh market data...")
            files = download_data(TOKEN, TRAINING_DAYS, REGION, DATA_PROVIDER)
        else:
            files = []

        # Force update to regenerate data
        format_data(files, DATA_PROVIDER, force_update=True)

        print("📊 Training Model...")
        train_model(TIMEFRAME)

    except Exception as e:
        print(f"❌ Update failed: {e}")

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(str(inference), status=200, mimetype='application/json')
    
    except FileNotFoundError:
        return Response(json.dumps({"error": "Model not found. Please update first."}), status=500, mimetype='application/json')

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return Response(json.dumps({"status": "success"}), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"status": "failed", "error": str(e)}), status=500, mimetype='application/json')
if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
