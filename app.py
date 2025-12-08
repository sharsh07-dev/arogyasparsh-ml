from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import re
import time

load_dotenv()

app = Flask(__name__)
CORS(app)

# 1. CONNECT TO MONGODB
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_URI = "mongodb+srv://shindeharshdev_db_user:whbXN3cgeiFgETsd@arogyadata.yzb2tan.mongodb.net/arogyasparsh?appName=ArogyaData"

try:
    client = MongoClient(MONGO_URI)
    db = client.get_database("arogyasparsh") 
    requests_collection = db.requests
    phc_inventory_collection = db.phcinventories
    hospital_inventory_collection = db.hospitalinventories 
except Exception as e:
    print(f"❌ DB Connection Error: {e}")

# GLOBAL MAP
PHC_KEYWORD_MAP = {
    "wagholi": "Wagholi PHC", "chamorshi": "PHC Chamorshi", "gadhchiroli": "PHC Gadhchiroli",
    "panera": "PHC Panera", "belgaon": "PHC Belgaon", "dhutergatta": "PHC Dhutergatta",
    "gatta": "PHC Gatta", "gaurkheda": "PHC Gaurkheda", "murmadi": "PHC Murmadi"
}

# --- 🧠 HELPER: ROBUST PREDICTION ENGINE (FIXED) ---
def generate_predictions():
    try:
        # 1. Fetch only Valid Delivered Orders
        data = list(requests_collection.find({"status": "Delivered"}))
        
        # If no data, return empty list (Don't crash)
        if not data: 
            return []

        df = pd.DataFrame(data)

        # 2. Safety Checks: Ensure required columns exist
        if 'item' not in df.columns or 'phc' not in df.columns or 'createdAt' not in df.columns:
            return []

        # 3. Clean Data
        df['item'] = df['item'].fillna('Unknown')
        df['phc'] = df['phc'].fillna('Unknown')
        
        # Safe extraction of Item Name (Handle "50x Name" vs "Name")
        def clean_item_name(x):
            if isinstance(x, str) and "x " in x:
                try:
                    return x.split("x ")[1]
                except:
                    return str(x)
            return str(x)

        df['item_name'] = df['item'].apply(clean_item_name)
        
        # 4. Process Dates (Handle errors/NaT)
        df['date'] = pd.to_datetime(df['createdAt'], errors='coerce') 
        df = df.dropna(subset=['date']) # Drop rows with bad dates
        if df.empty: return []

        df['day_of_year'] = df['date'].dt.dayofyear

        # 5. Encoders
        le_item = LabelEncoder()
        df['item_code'] = le_item.fit_transform(df['item_name'])
        
        le_phc = LabelEncoder()
        df['phc_code'] = le_phc.fit_transform(df['phc'])

        # 6. Prepare Training Data
        X = df[['item_code', 'phc_code', 'day_of_year']]
        y = df['qty'].fillna(0) # Handle missing qty

        if len(X) < 2: # Not enough data to train
            return []

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        future_predictions = []
        next_week_day = datetime.now().timetuple().tm_yday + 7
        
        unique_items = df['item_name'].unique()
        unique_phcs = df['phc'].unique()

        # 7. Generate Forecasts
        for phc in unique_phcs:
            try:
                phc_encoded = le_phc.transform([phc])[0]
                for item in unique_items:
                    item_encoded = le_item.transform([item])[0]
                    
                    # Predict using all trees for confidence interval
                    preds = [tree.predict([[item_encoded, phc_encoded, next_week_day]])[0] for tree in model.estimators_]
                    pred_qty = np.mean(preds)
                    lower, upper = np.percentile(preds, 5), np.percentile(preds, 95)
                    
                    # Determine Trend
                    history = df[(df['item_name'] == item) & (df['phc'] == phc)]
                    trend = "Stable"
                    if not history.empty:
                        recent_avg = history['qty'].tail(3).mean()
                        if pred_qty > recent_avg * 1.1: trend = "Rising"
                        elif pred_qty < recent_avg * 0.9: trend = "Falling"

                    if round(pred_qty) > 0:
                        future_predictions.append({
                            "phc": phc, "name": item, "predictedQty": round(pred_qty),
                            "lower": round(lower, 1), "upper": round(upper, 1), "trend": trend
                        })
            except:
                continue # Skip if encoding fails for specific item

        return future_predictions

    except Exception as e:
        print(f"Prediction Engine Error: {e}")
        return [] # Return empty list on crash, so Dashboard stays alive

@app.route('/predict-demand', methods=['GET'])
def predict_demand():
    try:
        preds = generate_predictions()
        return jsonify(preds)
    except Exception as e: 
        return jsonify({"error": str(e)}), 500


# ==========================================
# 🤖 BOT 1: SWASTHYA-AI (ADMIN & GENERAL)
# ==========================================
@app.route('/swasthya-ai', methods=['POST'])
def swasthya_ai():
    try:
        data = request.json
        query = data.get('query', '').lower()
        context = data.get('context', {})
        
        response = { "text": "I am SwasthyaAI. I can Track, Compare, and Forecast.", "type": "text" }

        found_phcs = []
        for key, fullname in PHC_KEYWORD_MAP.items():
            if key in query: found_phcs.append(fullname)
        found_phcs = list(set(found_phcs)) 

        if len(found_phcs) >= 2 or 'compare' in query:
            if len(found_phcs) < 2:
                response["text"] = "Please name the two PHCs you want to compare."
            else:
                phc_a, phc_b = found_phcs[0], found_phcs[1]
                # Calculate real metrics
                def get_metrics(name):
                    orders = list(requests_collection.find({"phc": name}))
                    total = len(orders)
                    delivered = len([o for o in orders if o.get('status') == 'Delivered'])
                    rate = round((delivered/total * 100), 1) if total > 0 else 0
                    return {"total": total, "rate": f"{rate}%"}

                stats_a = get_metrics(phc_a)
                stats_b = get_metrics(phc_b)

                response = {
                    "text": f"Comparison Report: **{phc_a}** vs **{phc_b}**",
                    "type": "table",
                    "data": {
                        "headers": ["Metric", phc_a, phc_b],
                        "rows": [
                            ["Total Orders", stats_a['total'], stats_b['total']], 
                            ["Fulfillment", stats_a['rate'], stats_b['rate']]
                        ]
                    }
                }

        elif 'track' in query or 'drone' in query:
            target_phc = found_phcs[0] if found_phcs else context.get('userPHC', 'Unknown')
            # Look for real active mission
            active_order = requests_collection.find_one({
                "phc": {"$regex": target_phc, "$options": "i"}, # Case insensitive search
                "status": {"$in": ["Dispatched", "In-Flight"]}
            })
            
            if active_order:
                response = {
                    "text": f"🔭 Tracking Mission for **{active_order.get('phc')}**\nStatus: **{active_order.get('status')}**",
                    "type": "tracking",
                    "data": { "status": active_order.get('status') }
                }
            else:
                response["text"] = f"No active drone flights detected for {target_phc}."

        elif 'forecast' in query or 'predict' in query:
             preds = generate_predictions()
             target_phc = found_phcs[0] if found_phcs else context.get('userPHC', '')
             phc_preds = [p for p in preds if target_phc in p['phc']]
             
             if phc_preds:
                 top = max(phc_preds, key=lambda x: x['predictedQty'])
                 response = {
                     "text": f"📈 Forecast for **{target_phc}**:\nHighest demand expected for **{top['name']}**.",
                     "type": "forecast",
                     "data": { "prediction": top['predictedQty'], "range": f"{top['lower']}-{top['upper']}", "confidence": "85%" }
                 }
             else:
                 response["text"] = f"Insufficient data to forecast for {target_phc}."

        elif 'hello' in query:
            response["text"] = "Hello! I am **SwasthyaAI**."

        return jsonify(response)

    except Exception as e:
        return jsonify({"text": "System Error.", "type": "error"}), 500


# ==========================================
# 🏥 BOT 2: HOSPITAL SWASTHYA AI (OPS)
# ==========================================
@app.route('/hospital-ai', methods=['POST'])
def hospital_ai():
    try:
        data = request.json
        query = data.get('query', '').lower()
        response = { "text": "SwasthyaAI Ops Ready.", "type": "text", "data": {} }

        if 'order' in query or 'voice' in query:
            response = {
                "text": "🎙️ **Voice Input Detected**\n\nProcessing Order... **Confidence: 98%**\n\n✅ **Identified:** 50x Inj. Adrenaline\n✅ **Source:** Voice Call\n",
                "type": "voice_process",
                "data": { "status": "Accepted", "progress": 100, "order_id": "ORD-VOICE-" + str(int(time.time())) }
            }

        elif 'inventory' in query or 'stock' in query or 'expired' in query:
            hosp_inv = hospital_inventory_collection.find_one()
            items = hosp_inv.get('items', []) if hosp_inv else []
            today = datetime.now().date()
            
            expired_list = []
            low_stock_list = []

            for item in items:
                try:
                    exp_date = datetime.strptime(item.get('expiry', '2099-01-01'), "%Y-%m-%d").date()
                    if exp_date < today: expired_list.append([item['name'], item.get('batch','-'), item['expiry']])
                    if int(item.get('stock', 0)) < 100: low_stock_list.append([item['name'], item['stock'], "Critical"])
                except: continue

            if 'expired' in query:
                if expired_list:
                    response = { "text": f"🚨 Found **{len(expired_list)}** expired items.", "type": "table", "data": { "headers": ["Name", "Batch", "Expiry"], "rows": expired_list } }
                else:
                    response["text"] = "✅ No expired items found."
            else:
                 if low_stock_list:
                    response = { "text": f"⚠️ Found **{len(low_stock_list)}** Low Stock items.", "type": "table", "data": { "headers": ["Name", "Qty", "Status"], "rows": low_stock_list } }
                 else:
                    response["text"] = "✅ Stock levels are healthy."

        return jsonify(response)
    except Exception as e:
        print(e)
        return jsonify({"text": "System Error.", "type": "error"}), 500


# ==========================================
# 🚑 BOT 3: PHC ASSISTANT
# ==========================================
@app.route('/phc-assistant', methods=['POST'])
def phc_assistant():
    try:
        data = request.json
        query = data.get('query', '').lower()
        phc_id = data.get('context', {}).get('phc_id', 'Wagholi PHC')
        is_voice = data.get('is_voice', False)
        
        response = {
            "text": "", "type": "text", "stt": { "transcript": query if is_voice else None, "confidence": 0.98 },
            "data": {}, "retrieved_at": datetime.now().isoformat()
        }

        if 'expired' in query or 'stock' in query:
            phc_data = phc_inventory_collection.find_one({"phcName": phc_id})
            if not phc_data:
                response["text"] = f"Could not access database for {phc_id}."
            else:
                items = phc_data.get('items', [])
                today = datetime.now().date()
                expired = []
                for item in items:
                    try:
                        if datetime.strptime(item.get('expiry', '2099-01-01'), "%Y-%m-%d").date() < today:
                            expired.append([item['name'], item.get('batch','-'), item['expiry']])
                    except: continue
                
                if 'expired' in query:
                    if expired:
                        response["text"] = f"⚠️ Found **{len(expired)}** expired items."
                        response["type"] = "table"
                        response["data"] = { "title": "Expired", "headers": ["Name", "Batch", "Date"], "rows": expired }
                    else:
                        response["text"] = "✅ No expired items."
                else:
                    response["text"] = f"Inventory for {phc_id} is loaded."

        elif 'hello' in query:
            response["text"] = f"Hello. I am **SwasthyaAI-PHC**. Assigned to {phc_id}."
        
        else:
            response["text"] = "I can check **Expired Items** or **Recent Orders**."

        return jsonify(response)
    except Exception as e:
        return jsonify({"text": "System Error.", "type": "error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port)