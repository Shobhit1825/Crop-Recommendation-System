from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
 
# Load models and scalers
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except FileNotFoundError as e:
    print("Error: One or more model files were not found. Check the file paths and ensure they are in the same directory as this script.")
    raise e
except Exception as e:
    print(f"An unexpected error occurred while loading the model files: {e}")
    raise e
 
# Initialize Flask app
app = Flask(__name__)
 
@app.route('/')
def index():
    return render_template("index.html")
 
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Collect input data and convert to floats
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
 
        # Prepare feature array
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
 
        # Apply scalers
        scaled_features = sc.transform(single_pred)
        final_features = ms.transform(scaled_features)
 
        # Predict crop type
        prediction = model.predict(final_features)
 
        # Crop dictionary
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
            11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
            15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
 
        # Get the crop recommendation
        crop = crop_dict.get(prediction[0], "Unknown")
        result = f"{crop} is the best crop to be cultivated right there."
 
    except Exception as e:
        result = "Error processing the prediction. Please check the input values and try again."
        print(f"Error: {e}")
 
    return render_template('index.html', result=result)
 
if __name__ == "__main__":
    app.run(debug=True)