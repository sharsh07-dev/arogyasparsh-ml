from flask import Flask, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
import datetime
import numpy as np

load_dotenv()

app = Flask(__name__)
CORS(app)

# 1. CONNECT TO MONGODB
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    # Fallback for local testing
    MONGO_URI = "mongodb+srv://shindeharshdev_db_user:whbXN3cgeiFgETsd@arogyadata.yzb2tan.mongodb.net/arogyasparsh?appName=ArogyaData"

client = MongoClient(MONGO_URI)
db = client.get_database("arogyasparsh") 
requests_collection = db.requests

@app.route('/predict-demand', methods=['GET'])
def predict_demand():
    try:
        # 2. FETCH DATA (Delivered Orders Only)
        data = list(requests_collection.find({"status": "Delivered"}))
        
        # If no data, return empty list (Frontend handles fallback)
        if not data:
            return jsonify([])

        df = pd.DataFrame(data)
        
        # 3. PREPROCESS DATA
        # Clean Item Names
        df['item_name'] = df['item'].apply(lambda x: x.split("x ")[1] if "x " in x else x)
        df['date'] = pd.to_datetime(df['createdAt'])
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Encode Categories (Text -> Numbers)
        le_item = LabelEncoder()
        df['item_code'] = le_item.fit_transform(df['item_name'])
        
        le_phc = LabelEncoder()
        df['phc_code'] = le_phc.fit_transform(df['phc'])

        # 4. TRAIN MODEL (Random Forest)
        # Features: [Item, PHC, Day] -> Target: [Quantity]
        X = df[['item_code', 'phc_code', 'day_of_year']]
        y = df['qty']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 5. GENERATE PREDICTIONS FOR NEXT WEEK
        future_predictions = []
        next_week_day = datetime.datetime.now().timetuple().tm_yday + 7
        
        unique_items = df['item_name'].unique()
        unique_phcs = df['phc'].unique()

        for phc in unique_phcs:
            phc_encoded = le_phc.transform([phc])[0]
            
            for item in unique_items:
                item_encoded = le_item.transform([item])[0]
                
                # Predict
                pred_qty = model.predict([[item_encoded, phc_encoded, next_week_day]])[0]
                
                # Calculate Trend
                # Filter history for this specific PHC + Item
                history = df[(df['item_name'] == item) & (df['phc'] == phc)]
                
                trend = "➡️ Stable"
                if not history.empty:
                    recent_avg = history['qty'].tail(3).mean()
                    if pred_qty > recent_avg * 1.1: trend = "📈 Rising"
                    elif pred_qty < recent_avg * 0.9: trend = "📉 Falling"

                if round(pred_qty) > 0:
                    future_predictions.append({
                        "phc": phc,  # ✅ Now includes PHC Name
                        "name": item,
                        "predictedQty": round(pred_qty),
                        "trend": trend
                    })

        return jsonify(future_predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port)