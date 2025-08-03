from flask import Flask, request, jsonify, render_template
import fastf1
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)

# Enable FastF1 Cache
fastf1.Cache.enable_cache("E:/python codes/cache")

#DYNAMIC MODEL LOADING#
# A dictionary of model paths keyed by (race, session)
# so you can pick the correct model automatically
MODEL_PATHS = {
    ("Bahrain", "R"): "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Bahrain Grand Prix_Race.keras",
    ("Belgian", "R"): "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Belgian Grand Prix_Race.keras",
    ("Austrain", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Austrian Grand Prix_Race.keras",
    ("Abu Dhabi","R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Abu Dhabi Grand Prix_Race.keras",
    ("British", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\British Grand Prix.keras",
    ("Canada", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Canadian Grand Prix.keras",
    ("Azerbaijan", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Azerbaijan Grand Prix.keras",
    ("Chinese", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Chinese Grand Prix.keras",
    ("Dutch", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Dutch Grand Prix.keras",
    ("Emilia Romagna","R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Emilia Romagna Grand Prix.keras",
    ("French", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\French Grand Prix.keras",
    ("Hungarian", "R") :  "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Hungarian Grand Prix.keras",
    ("Italian", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Italian Grand Prix.keras",
    ("Japanese", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Japanese Grand Prix.keras",
    ("Las Vegas", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Las Vegas Grand Prix.keras",
    ("Mexico City", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Mexico City Grand Prix.keras",
    ("Miami", "R") :  "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Miami Grand Prix.keras",
    ("Monaco","R") :  "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Monaco Grand Prix.keras",
    ("Portuguese","R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Portuguese Grand Prix.keras",
    ("Qatar", "R") :  "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Qatar Grand Prix.keras",
    ("Russian", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Russian Grand Prix.keras",
    ("Sao Paulo", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\São Paulo Grand Prix.keras",
    ("Saudi Arabian", "R") :  "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Saudi Arabian Grand Prix.keras",
    ("Singapore", "R") :  "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Singapore Grand Prix.keras",
    ("Spanish", "R") : "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Spanish Grand Prix.keras",
    ("United States", "R") : "E:\\HM\\F1_Laptime_Predictor-main\\HM\\models\\converted_models\\United States Grand Prix.keras"

}

def load_model_for(race, session_type):
    """
    Return a loaded model for the given year, race, session_type.
    Fallback or error if not found.
    """
    # Try to find the path in the dictionary
    key = (race, session_type)
    if key in MODEL_PATHS:
        path = MODEL_PATHS[key]
        return load_model(path, custom_objects={"LeakyReLU": LeakyReLU, "mse": MeanSquaredError()})
    else:
        # fallback or raise
        # for demonstration, just return your existing default model
        default_path = "E:\HM\F1_Laptime_Predictor-main\HM\models\converted_models\Bahrain Grand Prix_Race.keras"
        return load_model(default_path, custom_objects={"LeakyReLU": LeakyReLU, "mse": MeanSquaredError()})


#Serve index.html#
@app.route('/')
def home():
    return render_template("index.html")


#1) Predict Lap Times (with year/race/session)#
@app.route("/predict", methods=["POST"])
def predict_lap_times():
    """
    Expects JSON with:
      { 
        "start_lap": ..., 
        "num_laps": ..., 
        "compound": ..., 
        "yearPredict": 2024,
        "racePredict": "Belgian",
        "sessionPredict": "R"
      }
    """
    data = request.json
    start_lap = int(data["start_lap"])
    num_laps = int(data["num_laps"])
    compound = data["compound"]

    # Additional selection params
    race_predict = data.get("racePredict", "Belgian")
    session_predict = data.get("sessionPredict", "R")

    DEGRADATION_FACTORS = {
        "SOFT": (0.07, 2.0),
        "MEDIUM": (0.05, 1.5),
        "HARD": (0.03, 1.2)
    }

    # Load the correct model for the user's selection
    model = load_model_for(race_predict, session_predict)

    lap_times = []
    for lap in range(start_lap, start_lap + num_laps):
        k1, k2 = DEGRADATION_FACTORS[compound]
        degradation = k1 * np.log(k2 * lap + 1)
        fuel_load = max(110 - (lap * 1.5), 0)  #fuel load =110 fuel_burn_rate = 1.5perlap
        track_temp = 35 - (lap // 10)    #track_temp = 35, temp_decrease_laps=10
        # Example input features: [lap, degradation, fuel, track_temp]
        input_data = np.array([[lap, degradation, fuel_load, track_temp]], dtype=np.float32).reshape((1, 1, 4))
        predicted_time = model.predict(input_data, verbose=0)[0][0]

        # Convert predicted_time (seconds) -> "MM:SS.mmm"
        minutes, seconds = divmod(predicted_time, 60)
        formatted_time = f"{int(minutes):02}:{seconds:.3f}"

        lap_times.append({"lap": lap, "time": formatted_time})

    return jsonify({"lap_times": lap_times})


#2) Position Changes#
@app.route('/api/position_changes', methods=['GET'])
def api_position_changes():
    year = int(request.args.get('year'))
    race = request.args.get('race')
    session_type = request.args.get('session_type')

    # e.g. "Quali" in front end => "Q" in fastf1?
    # If your HTML <option value="Q">Quali</option>, that is already "Q", so we are good
    session = fastf1.get_session(year, race, session_type)
    session.load()

    laps = session.laps
    if laps.empty:
        return jsonify({"labels": [], "datasets": []})

    max_lap = int(laps['LapNumber'].max())
    lap_numbers = list(range(1, max_lap + 1))

    datasets = []
    for drv in session.drivers:
        driver_laps = laps.pick_driver(drv)
        if driver_laps.empty:
            continue
        driver_name = driver_laps['Driver'].iloc[0]

        positions = []
        for lap_num in lap_numbers:
            row = driver_laps[driver_laps['LapNumber'] == lap_num]
            if not row.empty:
                positions.append(int(row['Position']))
            else:
                positions.append(None)

        color_hex = "#" + ''.join(
            [hex((hash(driver_name) >> i) & 0xFF)[2:].zfill(2)
             for i in (0, 8, 16)]
        )

        datasets.append({
            "label": driver_name,
            "data": positions,
            "borderColor": color_hex,
            "fill": False
        })

    return jsonify({
        "labels": lap_numbers,
        "datasets": datasets
    })


#3) Driver Lap Times#
@app.route('/api/driver_laptimes', methods=['GET'])
def api_driver_laptimes():
    year = int(request.args.get('year'))
    race = request.args.get('race')
    session_type = request.args.get('session_type')
    driver = request.args.get('driver')

    session = fastf1.get_session(year, race, session_type)
    session.load()

    laps = session.laps.pick_driver(driver)
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    scatter_data = []
    for _, row in laps.iterrows():
        if pd.isna(row['LapTimeSeconds']):
            continue
        scatter_data.append({
            "x": int(row['LapNumber']),
            "y": float(row['LapTimeSeconds'])  # front-end will convert to min or format
        })

    return jsonify({
        "dataset": {
            "label": driver,
            "data": scatter_data,
            "showLine": True,
            "borderColor": "black",
            "backgroundColor": "blue"
        }
    })


#4) Lap Times Distribution#
@app.route('/api/laptimes_distribution', methods=['GET'])
def api_laptimes_distribution():
    year = int(request.args.get('year'))
    race = request.args.get('race')
    session_type = request.args.get('session_type')
    driver = request.args.get('driver')

    session = fastf1.get_session(year, race, session_type)
    session.load()

    laps = session.laps.pick_driver(driver)
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds().dropna()

    if laps['LapTimeSeconds'].empty:
        return jsonify({
            "binEdges": [],
            "binCounts": []
        })

    bin_counts, bin_edges = np.histogram(laps['LapTimeSeconds'], bins=10)

    return jsonify({
        "binEdges": bin_edges.round(2).tolist(),
        "binCounts": bin_counts.tolist()
    })


#5) Driver Styling#
@app.route('/api/driver_styling', methods=['GET'])
def api_driver_styling():
    year = int(request.args.get('year'))
    race = request.args.get('race')
    session_type = request.args.get('session_type')
    driver = request.args.get('driver')

    session = fastf1.get_session(year, race, session_type)
    session.load()

    laps = session.laps.pick_driver(driver)
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    labels = [f"Lap {int(ln)}" for ln in laps['LapNumber']]
    data_values = laps['LapTimeSeconds'].fillna(0).tolist()

    comp_colors = {
        'SOFT': 'rgba(255,0,0,0.6)',
        'MEDIUM': 'rgba(255,255,0,0.6)',
        'HARD': 'rgba(255,255,255,0.6)'
    }
    background_colors = []
    for idx, row in laps.iterrows():
        c = comp_colors.get(row['Compound'], 'rgba(100,100,100,0.6)')
        background_colors.append(c)

    dataset = {
        "label": "Lap Times by Compound",
        "data": data_values,  # still in sec, front-end will handle
        "backgroundColor": background_colors,
        "borderColor": "rgba(255,255,255,0.8)",
        "borderWidth": 1
    }

    return jsonify({
        "labels": labels,
        "dataset": dataset
    })


if __name__ == "__main__":
    app.run(debug=True)
