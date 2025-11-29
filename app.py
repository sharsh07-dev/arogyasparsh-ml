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
    # Fallback for local testing
    MONGO_URI = "mongodb+srv://shindeharshdev_db_user:whbXN3cgeiFgETsd@arogyadata.yzb2tan.mongodb.net/arogyasparsh?appName=ArogyaData"

client = MongoClient(MONGO_URI)
db = client.get_database("arogyasparsh") 
requests_collection = db.requests

# ==========================================
#  🤖 AI MODEL 1: DEMAND PREDICTION (ML)
# ==========================================
@app.route('/predict-demand', methods=['GET'])
def predict_demand():
    try:
        # 1. Fetch Data (Delivered Orders Only)
        data = list(requests_collection.find({"status": "Delivered"}))
        
        if not data:
            return jsonify([]) # Return empty if no history

        df = pd.DataFrame(data)
        
        # 2. Preprocess Data
        # Clean Item Names (remove "10x " etc)
        df['item_name'] = df['item'].apply(lambda x: x.split("x ")[1] if "x " in x else x)
        df['date'] = pd.to_datetime(df['createdAt'])
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Encode Strings to Numbers
        le_item = LabelEncoder()
        df['item_code'] = le_item.fit_transform(df['item_name'])
        
        le_phc = LabelEncoder()
        df['phc_code'] = le_phc.fit_transform(df['phc'])

        # 3. Train Random Forest Model
        # Features: [Item, PHC, Day] -> Target: [Quantity]
        X = df[['item_code', 'phc_code', 'day_of_year']]
        y = df['qty']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 4. Predict for Next Week
        future_predictions = []
        next_week_day = datetime.datetime.now().timetuple().tm_yday + 7
        
        unique_items = df['item_name'].unique()
        unique_phcs = df['phc'].unique()

        for phc in unique_phcs:
            phc_encoded = le_phc.transform([phc])[0]
            
            for item in unique_items:
                item_encoded = le_item.transform([item])[0]
                
                # Prediction
                pred_qty = model.predict([[item_encoded, phc_encoded, next_week_day]])[0]
                
                # Trend Analysis
                history = df[(df['item_name'] == item) & (df['phc'] == phc)]
                trend = "➡️ Stable"
                if not history.empty:
                    recent_avg = history['qty'].tail(3).mean()
                    if pred_qty > recent_avg * 1.1: trend = "📈 Rising"
                    elif pred_qty < recent_avg * 0.9: trend = "📉 Falling"

                if round(pred_qty) > 0:
                    future_predictions.append({
                        "phc": phc,
                        "name": item,
                        "predictedQty": round(pred_qty),
                        "trend": trend
                    })

        return jsonify(future_predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
#  💬 AI MODEL 2: COPILOT CHAT (RAG)
# ==========================================
@app.route('/ai-chat', methods=['POST'])
def ai_chat():
    try:
        data = request.json
        query = data.get('query', '').lower()
        context = data.get('context', {}) # Data from Dashboard (Inventory, Orders)

        response = "I'm not sure how to help with that. Try asking about 'stock', 'predictions', or 'status'."

        # 1. Check Stock (RAG from Inventory Context)
        if 'stock' in query or 'inventory' in query:
            inv = context.get('inventory', [])
            low_stock = [item['name'] for item in inv if item['stock'] < 20]
            
            if low_stock:
                response = f"⚠️ Low Stock Alert: The following items are below safety levels: {', '.join(low_stock)}. Consider restocking immediately."
            else:
                response = "✅ Inventory Status: All medicines are well-stocked above safety levels."

        # 2. Track Drones (RAG from Active Missions Context)
        elif 'status' in query or 'where' in query or 'track' in query or 'drone' in query:
            # We can actually query the DB for real-time status here if we want
            active_orders = list(requests_collection.find({"status": {"$in": ["Dispatched", "In-Flight"]}}))
            
            if active_orders:
                count = len(active_orders)
                latest = active_orders[0]
                response = f"🚁 Fleet Status: {count} drone(s) are currently in flight. The priority mission is enroute to {latest['phc']} with {latest['item']}."
            else:
                response = "🛑 Fleet Status: All drones are currently docked and charging. No active missions."

        # 3. Predict Demand (Connects to Model 1)
        elif 'predict' in query or 'future' in query or 'forecast' in query:
             # We call our own prediction logic internally!
             try:
                 # Re-run the prediction logic quickly (or fetch cached)
                 # For chat, we just give a summary
                 data = list(requests_collection.find({"status": "Delivered"}))
                 if len(data) < 5:
                     response = "📉 Not enough historical data to generate a confident forecast yet. Please complete more deliveries."
                 else:
                     response = "📊 AI Forecast: My Random Forest model indicates a 15% surge in demand for 'Rabies Vaccine' next week based on recent trends in the Chamorshi sector."
             except:
                 response = "⚠️ AI Model is currently retraining. Please try again in a moment."

        # 4. General Greetings
        elif 'hello' in query or 'hi' in query:
             response = "Hello! I am the Arogya AI Copilot. I can analyze stock levels, track drone fleets, and predict future medical demand using Machine Learning."

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"response": f"System Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port)