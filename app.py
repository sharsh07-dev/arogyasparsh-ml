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
inventory_collection = db.phcinventories # ✅ NEW: Link to PHC Stock DB

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

# --- ✅ INTELLIGENT AI COPILOT ---
@app.route('/ai-chat', methods=['POST'])
def ai_chat():
    try:
        data = request.json
        query = data.get('query', '').lower()
        context = data.get('context', {}) 

        response = "I'm not sure how to help with that. Try asking about 'stock', 'predictions', or 'status of Wagholi'."

        # 1. ✅ CHECK STOCK (Global or Specific PHC)
        if 'stock' in query or 'inventory' in query:
            # Check if a specific PHC name is mentioned
            target_phc = None
            # Map common names to DB names
            phc_map = {
                "wagholi": "Wagholi PHC",
                "chamorshi": "PHC Chamorshi",
                "gadhchiroli": "PHC Gadhchiroli",
                "panera": "PHC Panera",
                "belgaon": "PHC Belgaon",
                "dhutergatta": "PHC Dhutergatta",
                "gatta": "PHC Gatta",
                "gaurkheda": "PHC Gaurkheda",
                "murmadi": "PHC Murmadi"
            }
            
            for key in phc_map:
                if key in query:
                    target_phc = phc_map[key]
                    break
            
            if target_phc:
                # Fetch remote data from MongoDB
                phc_data = inventory_collection.find_one({"phcName": target_phc})
                if phc_data:
                    items = phc_data.get('items', [])
                    low_stock = [i['name'] for i in items if i['stock'] < 20]
                    if low_stock:
                        response = f"⚠️ REMOTE ALERT ({target_phc}): Critical low stock detected for: {', '.join(low_stock)}."
                    else:
                        response = f"✅ Status Good: {target_phc} has healthy stock levels for all essential medicines."
                else:
                    response = f"ℹ️ No digital inventory found for {target_phc}. They may not have initialized their stock dashboard yet."
            
            else:
                # Fallback: Check LOCAL context (The dashboard user is looking at)
                inv = context.get('inventory', [])
                if not inv:
                    response = "I can't see any inventory here. Please specify a PHC name (e.g., 'Stock at Wagholi')."
                else:
                    low_stock = [item['name'] for item in inv if item['stock'] < 20]
                    if low_stock:
                        response = f"⚠️ Low Stock Alert (Current Location): The following items are below safety levels: {', '.join(low_stock)}."
                    else:
                        response = "✅ Current Inventory Status: All items are well-stocked."

        # 2. Track Drones (Dynamic)
        elif 'status' in query or 'where' in query or 'track' in query or 'drone' in query:
            active_orders = list(requests_collection.find({"status": {"$in": ["Dispatched", "In-Flight", "Delivered"]}}))
            
            target_phc = None
            phc_list = ["wagholi", "chamorshi", "gadhchiroli", "panera", "belgaon", "dhutergatta", "gatta", "gaurkheda", "murmadi"]
            for p in phc_list:
                if p in query: target_phc = p; break
            
            if target_phc:
                specific_mission = next((r for r in reversed(active_orders) if target_phc in r['phc'].lower()), None)
                if specific_mission:
                    status = specific_mission['status']
                    response = f"🚁 Update for {specific_mission['phc']}: Status is '{status}'. Cargo: {specific_mission['item']}."
                else:
                    response = f"ℹ️ No active flights found for {target_phc.title()}."
            else:
                in_flight = len([r for r in active_orders if r['status'] == 'In-Flight'])
                response = f"🚁 Fleet Status: {in_flight} drones currently airborne. Ask 'Track Panera' for details."

        # 3. Predict Demand
        elif 'predict' in query or 'future' in query or 'forecast' in query:
             try:
                 preds = generate_predictions()
                 if not preds:
                     response = "📉 Not enough data for predictions."
                 else:
                     # ... (Keep prediction filtering logic) ...
                     target_phc = None
                     for p in ["wagholi", "chamorshi", "gadhchiroli", "panera", "belgaon", "dhutergatta", "gatta", "gaurkheda", "murmadi"]:
                         if p in query: target_phc = p; break
                     
                     if target_phc:
                         phc_preds = [p for p in preds if target_phc in p['phc'].lower()]
                         if phc_preds:
                             top = max(phc_preds, key=lambda x: x['predictedQty'])
                             response = f"📊 AI Forecast ({top['phc']}): Expect demand of {top['predictedQty']} units for '{top['name']}' next week."
                         else:
                             response = f"ℹ️ Stable demand predicted for {target_phc}."
                     else:
                         top = max(preds, key=lambda x: x['predictedQty'])
                         response = f"📊 System-Wide Forecast: Highest surge expected at {top['phc']} ({top['predictedQty']} units of {top['name']})."
             except Exception as e:
                 response = f"⚠️ AI Error: {str(e)}"

        elif 'hello' in query or 'hi' in query:
             response = "Hello! I am the Arogya AI. I can check remote PHC stock, track drones, and predict medicine demand."

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"System Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port)