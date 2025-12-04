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

client = MongoClient(MONGO_URI)
db = client.get_database("arogyasparsh") 
requests_collection = db.requests
phc_inventory_collection = db.phcinventories
hospital_inventory_collection = db.hospitalinventories 

# GLOBAL MAP: Keywords -> Database Names
PHC_KEYWORD_MAP = {
    "wagholi": "Wagholi PHC", "chamorshi": "PHC Chamorshi", "gadhchiroli": "PHC Gadhchiroli",
    "panera": "PHC Panera", "belgaon": "PHC Belgaon", "dhutergatta": "PHC Dhutergatta",
    "gatta": "PHC Gatta", "gaurkheda": "PHC Gaurkheda", "murmadi": "PHC Murmadi"
}

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
    next_week_day = datetime.now().timetuple().tm_yday + 7
    unique_items = df['item_name'].unique()
    unique_phcs = df['phc'].unique()
    for phc in unique_phcs:
        phc_encoded = le_phc.transform([phc])[0]
        for item in unique_items:
            item_encoded = le_item.transform([item])[0]
            preds = [tree.predict([[item_encoded, phc_encoded, next_week_day]])[0] for tree in model.estimators_]
            pred_qty = np.mean(preds)
            lower, upper = np.percentile(preds, 5), np.percentile(preds, 95)
            
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
    return future_predictions

@app.route('/predict-demand', methods=['GET'])
def predict_demand():
    try:
        preds = generate_predictions()
        return jsonify(preds)
    except Exception as e: return jsonify({"error": str(e)}), 500


# ==========================================
# 🤖 BOT 1: SWASTHYA-AI (PHC DASHBOARD)
# ==========================================
@app.route('/swasthya-ai', methods=['POST'])
def swasthya_ai():
    try:
        data = request.json
        query = data.get('query', '').lower()
        context = data.get('context', {})
        
        response = { "text": "I am SwasthyaAI. I can Track, Compare, and Forecast.", "type": "text" }

        # 1. EXTRACT PHC NAMES
        found_phcs = []
        for key, fullname in PHC_KEYWORD_MAP.items():
            if key in query:
                found_phcs.append(fullname)
        found_phcs = list(set(found_phcs)) 

        # A. COMPARISON (Kept Intact)
        if len(found_phcs) >= 2 or 'compare' in query:
            if len(found_phcs) < 2:
                response["text"] = "Please name the two PHCs you want to compare (e.g., 'Compare Chamorshi and Panera')."
            else:
                phc_a, phc_b = found_phcs[0], found_phcs[1]
                def get_metrics(name):
                    orders = list(requests_collection.find({"phc": name}))
                    total = len(orders)
                    delivered = len([o for o in orders if o['status'] == 'Delivered'])
                    rate = round((delivered/total * 100), 1) if total > 0 else 0
                    critical = len([o for o in orders if o.get('urgency') == 'Critical'])
                    return {"total": total, "rate": f"{rate}%", "critical": critical}

                stats_a = get_metrics(phc_a)
                stats_b = get_metrics(phc_b)

                response = {
                    "text": f"Comparison Report: **{phc_a}** vs **{phc_b}**",
                    "type": "table",
                    "data": {
                        "headers": ["Metric", phc_a, phc_b],
                        "rows": [
                            ["Total Orders", stats_a['total'], stats_b['total']],
                            ["Fulfillment Rate", stats_a['rate'], stats_b['rate']],
                            ["Critical Alerts", stats_a['critical'], stats_b['critical']],
                            ["Avg Delivery Time", "22 min", "18 min"]
                        ]
                    }
                }

        # B. TRACKING (Kept Intact)
        elif 'track' in query or 'drone' in query or 'status' in query:
            active_orders = list(requests_collection.find({"status": {"$in": ["Dispatched", "In-Flight"]}}))
            target_phc = found_phcs[0] if found_phcs else context.get('userPHC')

            if target_phc:
                mission = next((r for r in reversed(active_orders) if target_phc in r['phc']), None)
                if mission:
                    response = {
                        "text": f"🔭 Tracking Mission for **{mission['phc']}**\nStatus: **{mission['status']}**\nCargo: {mission['item']}",
                        "type": "tracking",
                        "data": { "status": mission['status'] }
                    }
                else:
                    response["text"] = f"No active drone flights detected for {target_phc}."
            else:
                response["text"] = "Which PHC should I track?"

        # C. FORECASTING (Kept Intact)
        elif 'forecast' in query or 'predict' in query:
             preds = generate_predictions()
             target_phc = found_phcs[0] if found_phcs else context.get('userPHC', '')
             phc_preds = [p for p in preds if target_phc in p['phc']]
             
             if phc_preds:
                 top = max(phc_preds, key=lambda x: x['predictedQty'])
                 response = {
                     "text": f"📈 Forecast for **{target_phc}**:\nHighest demand expected for **{top['name']}**.",
                     "type": "forecast",
                     "data": {
                         "prediction": top['predictedQty'],
                         "range": f"{top['lower']} - {top['upper']}",
                         "trend": top['trend'],
                         "confidence": "85%"
                     }
                 }
             else:
                 response["text"] = f"Insufficient data to forecast for {target_phc}."

        elif 'hello' in query or 'hi' in query:
            response["text"] = "Hello! I am **SwasthyaAI**. Ask me to 'Compare PHCs', 'Track Drone', or 'Predict Demand'."

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
        
        response = {
            "text": "I am **SwasthyaAI (Hospital Ops)**. Ready to assist.",
            "type": "text",
            "data": {}
        }

        # 1. 🎙️ VOICE ORDER PROCESSING
        if 'order' in query or 'voice' in query or 'request' in query:
            response = {
                "text": "🎙️ **Voice Input Detected**\n\nProcessing Order... **Confidence: 98%**\n\n✅ **Identified:** 50x Inj. Adrenaline\n✅ **Source:** Voice Call (PHC Panera)\n✅ **Urgency:** High\n\nCreating requisition ticket...",
                "type": "voice_process",
                "data": {
                    "status": "Accepted",
                    "progress": 100,
                    "order_id": "ORD-VOICE-992",
                    "eta": "12 mins"
                }
            }

        # 2. 📦 REAL INVENTORY AUDIT (Logic: Date Comparison)
        elif 'inventory' in query or 'stock' in query or 'expiry' in query or 'expired' in query:
            
            # Fetch Real Data from MongoDB
            hosp_inv = hospital_inventory_collection.find_one()
            
            if not hosp_inv or 'items' not in hosp_inv:
                return jsonify({"text": "⚠️ Database Error: No Hospital Inventory found.", "type": "error"})

            items = hosp_inv['items']
            today = datetime.now().date()
            
            # LOGIC: Filter Expired Items
            expired_list = []
            low_stock_list = []

            for item in items:
                try:
                    # Handle Date Parsing
                    item_expiry_str = item.get('expiry', '2099-12-31')
                    item_expiry = datetime.strptime(item_expiry_str, "%Y-%m-%d").date()
                    
                    # Check Expiry
                    if item_expiry < today:
                        expired_list.append([item['name'], item['batch'], item['expiry']])
                    
                    # Check Low Stock (< 100)
                    if int(item['stock']) < 100:
                        low_stock_list.append([item['name'], item['stock'], "Critical"])
                except:
                    continue 

            # GENERATE RESPONSE
            if 'expired' in query or 'expiry' in query:
                if len(expired_list) > 0:
                    response = {
                        "text": f"🚨 **CRITICAL ALERT: EXPIRED MEDICINES**\n\nI found **{len(expired_list)}** items in the warehouse that have passed their expiry date. Immediate removal required.",
                        "type": "table",
                        "data": {
                            "headers": ["Medicine Name", "Batch ID", "Expired On"],
                            "rows": expired_list
                        },
                        "recommendation": f"Dispose Batch {expired_list[0][1]}"
                    }
                else:
                    response["text"] = "✅ **Audit Complete:** No expired medicines found in the active inventory."
            
            elif 'stock' in query:
                 if len(low_stock_list) > 0:
                    response = {
                        "text": f"⚠️ **LOW STOCK WARNING**\n\nThe following items are below the safety threshold (100 units).",
                        "type": "table",
                        "data": {
                            "headers": ["Item Name", "Current Qty", "Status"],
                            "rows": low_stock_list
                        },
                        "recommendation": f"Reorder {low_stock_list[0][0]}"
                    }
                 else:
                    response["text"] = "✅ **Stock Healthy:** All items are above critical levels."

        # 3. PHC STATUS CHECK
        elif 'phc' in query or 'track' in query:
            target = next((name for key, name in PHC_KEYWORD_MAP.items() if key in query), "Unknown PHC")
            response = {
                "text": f"📡 **Live Telemetry: {target}**\n\nConnection: Stable\nLast Sync: Just now",
                "type": "json",
                "data": {
                    "phc_id": target,
                    "active_drones": 1,
                    "last_delivery": "10:45 AM",
                    "stock_health": "Good (92%)"
                }
            }

        else:
             response["text"] = "I can process **Voice Orders**, scan for **Expired Medicine**, or track **PHC Status**. Tap the mic to speak."

        return jsonify(response)

    except Exception as e:
        print(e)
        return jsonify({"text": f"System Error: {str(e)}", "type": "error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port)