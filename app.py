from flask import Flask, jsonify, request
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
    MONGO_URI = "mongodb+srv://shindeharshdev_db_user:whbXN3cgeiFgETsd@arogyadata.yzb2tan.mongodb.net/arogyasparsh?appName=ArogyaData"

client = MongoClient(MONGO_URI)
db = client.get_database("arogyasparsh") 
requests_collection = db.requests

# --- HELPER: GENERATE PREDICTIONS ---
def generate_predictions():
    data = list(requests_collection.find({"status": "Delivered"}))
    if not data: return []
    df = pd.DataFrame(data)
    df['item_name'] = df['item'].apply(lambda x: x.split("x ")[1] if "x " in x else x)
    df['date'] = pd.to_datetime(df['createdAt'])
    df['day_of_year'] = df['date'].dt.dayofyear
    le_item = LabelEncoder()
    df['item_code'] = le_item.fit_transform(df['item_name'])
    le_phc = LabelEncoder()
    df['phc_code'] = le_phc.fit_transform(df['phc'])
    X = df[['item_code', 'phc_code', 'day_of_year']]
    y = df['qty']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    future_predictions = []
    next_week_day = datetime.datetime.now().timetuple().tm_yday + 7
    unique_items = df['item_name'].unique()
    unique_phcs = df['phc'].unique()
    for phc in unique_phcs:
        phc_encoded = le_phc.transform([phc])[0]
        for item in unique_items:
            item_encoded = le_item.transform([item])[0]
            pred_qty = model.predict([[item_encoded, phc_encoded, next_week_day]])[0]
            history = df[(df['item_name'] == item) & (df['phc'] == phc)]
            trend = "➡️ Stable"
            if not history.empty:
                recent_avg = history['qty'].tail(3).mean()
                if pred_qty > recent_avg * 1.1: trend = "📈 Rising"
                elif pred_qty < recent_avg * 0.9: trend = "📉 Falling"
            if round(pred_qty) > 0:
                future_predictions.append({"phc": phc, "name": item, "predictedQty": round(pred_qty), "trend": trend})
    return future_predictions

@app.route('/predict-demand', methods=['GET'])
def predict_demand():
    try:
        preds = generate_predictions()
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ✅ SMART AI CHATBOT (DYNAMIC) ---
@app.route('/ai-chat', methods=['POST'])
def ai_chat():
    try:
        data = request.json
        query = data.get('query', '').lower()
        context = data.get('context', {}) 

        response = "I'm not sure how to help with that. Try asking about 'stock', 'predictions', or 'status of Wagholi'."

        # 1. Check Stock (Dynamic from Context)
        if 'stock' in query or 'inventory' in query:
            inv = context.get('inventory', [])
            low_stock = [item['name'] for item in inv if item['stock'] < 20]
            if low_stock:
                response = f"⚠️ Low Stock Alert: {', '.join(low_stock)} are running low. Please restock."
            else:
                response = "✅ Inventory Status: All medicines are well-stocked."

        # 2. ✅ SMART TRACKING (Filters by PHC Name)
        elif 'status' in query or 'where' in query or 'track' in query or 'drone' in query:
            # Get ALL active missions from DB
            active_orders = list(requests_collection.find({"status": {"$in": ["Dispatched", "In-Flight", "Delivered"]}}))
            
            # Extract PHC Name from User Query (e.g., "Track Wagholi")
            target_phc = None
            phc_list = ["wagholi", "chamorshi", "gadhchiroli", "panera", "belgaon", "dhutergatta", "gatta", "gaurkheda", "murmadi"]
            
            for p in phc_list:
                if p in query:
                    target_phc = p
                    break
            
            if target_phc:
                # Find latest mission for THIS specific PHC
                # Sort by time to get the newest one
                specific_mission = next((r for r in reversed(active_orders) if target_phc in r['phc'].lower()), None)
                
                if specific_mission:
                    status_emoji = "✅" if specific_mission['status'] == "Delivered" else "🚁"
                    response = f"{status_emoji} Update for {specific_mission['phc']}: Latest order is '{specific_mission['status']}' carrying {specific_mission['item']}."
                else:
                    response = f"ℹ️ No recent drone activity found for {target_phc.title()} PHC."
            else:
                # General Summary
                in_flight_count = len([r for r in active_orders if r['status'] == 'In-Flight'])
                response = f"🚁 Fleet Overview: {in_flight_count} drones are currently in the air. Ask me to 'Track Wagholi' for specifics."

        # 3. Predict Demand (Dynamic by PHC)
        elif 'predict' in query or 'future' in query or 'forecast' in query:
             try:
                 preds = generate_predictions()
                 if not preds:
                     response = "📉 Not enough data for predictions yet."
                 else:
                     target_phc = None
                     for p in ["wagholi", "chamorshi", "gadhchiroli", "panera", "belgaon", "dhutergatta", "gatta", "gaurkheda", "murmadi"]:
                         if p in query:
                             target_phc = p
                             break
                     
                     if target_phc:
                         phc_preds = [p for p in preds if target_phc in p['phc'].lower()]
                         if phc_preds:
                             top = max(phc_preds, key=lambda x: x['predictedQty'])
                             response = f"📊 Forecast for {top['phc']}: Expect high demand for '{top['name']}' (~{top['predictedQty']} units) next week."
                         else:
                             response = f"ℹ️ No major demand spikes predicted for {target_phc}."
                     else:
                         top = max(preds, key=lambda x: x['predictedQty'])
                         response = f"📊 System Forecast: Highest network demand is at {top['phc']} for '{top['name']}'."
             except Exception as e:
                 response = f"⚠️ AI Model Error: {str(e)}"

        elif 'hello' in query or 'hi' in query:
             response = "Hello! I am the Arogya AI. Ask me to 'Track Panera', 'Check Stock', or 'Predict demand'."

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"System Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port)