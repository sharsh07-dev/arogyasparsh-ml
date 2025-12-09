"""
Enhanced SwasthyaAI Flask app
- More production-ready features while keeping single-file simplicity.
- Features added:
  * API-key based auth for sensitive endpoints
  * Structured logging
  * Model training, saving, loading (joblib)
  * Better feature engineering (day_of_year, week_of_year, month, day_of_week, rolling averages)
  * Per-PHC aggregation, district aggregation
  * Anomaly detection (z-score) on order quantities
  * Alert endpoints for low-stock & expiry and webhook support (placeholder)
  * Inventory CRUD endpoints
  * Retrain / incremental train endpoints
  * OpenAPI minimal spec at /openapi.json
  * Pagination for list endpoints
  * Safe fallbacks for scarce data
  * Configurable thresholds via environment variables
  * Health & metrics endpoint

Notes:
- This file assumes MongoDB and environment variables (MONGO_URI, API_KEY) are set.
- Model artifacts saved under ./models (encoders + scaler + model).
- No external scheduler is used; retraining is triggered via endpoints.

"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock

from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ------------------------
# Basic config & logging
# ------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("SwasthyaAI")

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

API_KEY = os.getenv("API_KEY", "changeme123")
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    MONGO_URI = "mongodb+srv://shindeharshdev_db_user:whbXN3cgeiFgETsd@arogyadata.yzb2tan.mongodb.net/arogyasparsh?appName=ArogyaData"

# Tunables
LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "100"))
ANOMALY_ZSCORE = float(os.getenv("ANOMALY_ZSCORE", "3.0"))
MIN_ROWS_FOR_ML = int(os.getenv("MIN_ROWS_FOR_ML", "8"))

app = Flask(__name__)
CORS(app)

# ------------------------
# DB connection
# ------------------------
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database("arogyasparsh")
    requests_collection = db.requests
    phc_inventory_collection = db.phcinventories
    hospital_inventory_collection = db.hospitalinventories
    logger.info("MongoDB connected")
except Exception as e:
    logger.exception("DB Connection Error: %s", e)
    # keep variables but they may be None in testing

# ------------------------
# In-memory artifacts + thread safety
# ------------------------
model_lock = Lock()
model = None
encoders = {"item": LabelEncoder(), "phc": LabelEncoder()}
scaler = None

# Try to load existing artifacts
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)
        logger.info("Loaded encoders")
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        logger.info("Loaded scaler")
except Exception as e:
    logger.exception("Failed loading artifacts: %s", e)

# ------------------------
# Helpers
# ------------------------

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key") or request.args.get("api_key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def df_from_requests(filter_query=None, limit=None):
    # Pull rows from MongoDB and return DataFrame with safe columns
    q = filter_query or {}
    cursor = requests_collection.find(q)
    if limit:
        cursor = cursor.limit(limit)
    data = list(cursor)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # safe defaults
    for c in ['item', 'phc', 'qty', 'createdAt']:
        if c not in df.columns:
            df[c] = None
    # normalize
    df['item'] = df['item'].fillna('Unknown').astype(str)
    df['phc'] = df['phc'].fillna('Unknown').astype(str)
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0)
    df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce').fillna(pd.Timestamp.now())
    return df


def feature_engineer(df):
    # Add time features
    df = df.copy()
    df['day_of_year'] = df['createdAt'].dt.dayofyear
    df['week_of_year'] = df['createdAt'].dt.isocalendar().week.astype(int)
    df['month'] = df['createdAt'].dt.month
    df['day_of_week'] = df['createdAt'].dt.dayofweek
    # Item name extraction: handle "10x Paracetamol" style or other patterns
    def clean_item(x):
        x = str(x)
        # heuristic: if contains 'x ' take right part
        if 'x ' in x:
            try:
                return x.split('x ', 1)[1].strip()
            except:
                return x
        return x
    df['item_name'] = df['item'].apply(clean_item)
    return df


def train_model_from_df(df):
    """Train RandomForest on given df and persist artifacts."""
    global model, encoders, scaler
    df = feature_engineer(df)
    # Use only meaningful rows
    Xdf = df[['item_name', 'phc', 'day_of_year', 'week_of_year', 'month', 'day_of_week']].copy()
    y = df['qty'].astype(float)

    # fit encoders on combined known categories
    encoders['item'] = LabelEncoder()
    encoders['phc'] = LabelEncoder()
    try:
        Xdf['item_code'] = encoders['item'].fit_transform(Xdf['item_name'])
        Xdf['phc_code'] = encoders['phc'].fit_transform(Xdf['phc'])
    except Exception as e:
        logger.exception("Encoder fit failed: %s", e)
        return None

    X = Xdf[['item_code', 'phc_code', 'day_of_year', 'week_of_year', 'month', 'day_of_week']].values

    # scaling day features for better RF behavior (optional)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # simple fallback guard
    if len(X_scaled) < MIN_ROWS_FOR_ML:
        logger.warning("Not enough rows (%d) to train ML; returning None", len(X_scaled))
        return None

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    with model_lock:
        model = rf
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Model and artifacts persisted")
    return model


def predict_for_next_period(target_phc=None, days_ahead=7, top_k=10):
    """Return predictions list of dicts for PHC(s) specified. If model missing, use averages fallback."""
    df = df_from_requests()
    if df.empty:
        return []
    df = feature_engineer(df)
    # unique items & phcs
    unique_items = df['item_name'].unique()
    unique_phcs = df['phc'].unique() if target_phc is None else [target_phc]

    results = []
    # compute next period features
    next_date = pd.Timestamp.now() + pd.Timedelta(days=days_ahead)
    day_of_year = int(next_date.dayofyear)
    week_of_year = int(next_date.isocalendar().week)
    month = int(next_date.month)
    day_of_week = int(next_date.dayofweek)

    # anomaly detection per item-phc historical: compute z-score and flag outliers
    grouped = df.groupby(['phc', 'item_name'])['qty']
    stats = grouped.agg(['mean', 'std', 'count']).reset_index()
    stats['std'] = stats['std'].replace(0, np.nan)

    for phc in unique_phcs:
        phc_subset = df[df['phc'] == phc]
        if phc_subset.empty:
            continue
        for item in unique_items:
            hist = phc_subset[phc_subset['item_name'] == item]
            if hist.empty:
                continue
            # fallback average
            avg = hist['qty'].mean()
            predicted = 0
            method = 'fallback_avg'

            # ML path
            if model is not None and os.path.exists(ENCODERS_PATH) and os.path.exists(SCALER_PATH):
                try:
                    item_code = encoders['item'].transform([item])[0]
                    phc_code = encoders['phc'].transform([phc])[0]
                    feat = np.array([[item_code, phc_code, day_of_year, week_of_year, month, day_of_week]])
                    feat = scaler.transform(feat)
                    predicted = float(model.predict(feat)[0])
                    method = 'ml'
                except Exception:
                    predicted = avg
                    method = 'fallback_avg'
            else:
                predicted = avg

            # post-process
            predicted = max(0, float(predicted))
            # detect anomaly in history (last observation)
            last_qty = hist['qty'].iloc[-1]
            st = stats[(stats['phc'] == phc) & (stats['item_name'] == item)]
            zscore = None
            anomalous = False
            if not st.empty:
                mu = st['mean'].values[0]
                sigma = st['std'].values[0] if not np.isnan(st['std'].values[0]) else 0
                if sigma > 0:
                    zscore = (last_qty - mu) / sigma
                    anomalous = abs(zscore) >= ANOMALY_ZSCORE

            results.append({
                'phc': phc,
                'item': item,
                'predictedQty': int(round(predicted)),
                'method': method,
                'recentAvg': int(round(avg)),
                'lastQty': int(round(float(last_qty))),
                'anomaly': anomalous,
                'zscore': None if zscore is None else float(zscore)
            })

    # aggregate district-level totals for convenience
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        district_agg = df_res.groupby('item')['predictedQty'].sum().reset_index()
        for _, row in district_agg.iterrows():
            results.append({
                'phc': 'District Aggregated',
                'item': row['item'],
                'predictedQty': int(row['predictedQty']),
                'method': 'aggregated',
                'recentAvg': None,
                'lastQty': None,
                'anomaly': False,
                'zscore': None
            })

    # return top_k by predictedQty
    results = sorted(results, key=lambda r: r['predictedQty'], reverse=True)
    return results[:top_k]

# ------------------------
# Endpoints
# ------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'time': datetime.utcnow().isoformat() + 'Z',
        'model_loaded': os.path.exists(MODEL_PATH)
    })

@app.route('/predict-demand', methods=['GET'])
@require_api_key
def predict_demand():
    phc = request.args.get('phc')
    days = int(request.args.get('days', '7'))
    topk = int(request.args.get('topk', '20'))
    try:
        preds = predict_for_next_period(target_phc=phc, days_ahead=days, top_k=topk)
        return jsonify(preds)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
@require_api_key
def train():
    """Trigger a full retrain on all data in DB. Returns status & stats."""
    try:
        df = df_from_requests()
        if df.empty:
            return jsonify({'status': 'no_data'}), 400
        m = train_model_from_df(df)
        if m is None:
            return jsonify({'status': 'insufficient_data', 'rows': len(df)}), 400
        return jsonify({'status': 'trained', 'rows': len(df)})
    except Exception as e:
        logger.exception("Train failed: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/train/incremental', methods=['POST'])
@require_api_key
def incremental_train():
    """Accept a payload of new orders and perform incremental fit-like behavior by retraining on union of data."""
    try:
        payload = request.json or {}
        new_orders = payload.get('orders', [])
        if not new_orders:
            return jsonify({'status': 'no_new_orders'}), 400
        # insert to DB (with basic cleaning)
        for o in new_orders:
            # minimal required fields
            o.setdefault('createdAt', datetime.utcnow().isoformat())
            requests_collection.insert_one(o)
        # retrain on all data (cheap for small datasets)
        df = df_from_requests()
        m = train_model_from_df(df)
        if m is None:
            return jsonify({'status': 'insufficient_data_after_insert'}), 400
        return jsonify({'status': 'incremental_trained', 'new_orders': len(new_orders)})
    except Exception as e:
        logger.exception("Incremental train failed: %s", e)
        return jsonify({'error': str(e)}), 500

# Inventory management endpoints
@app.route('/inventory/phc/<phc_name>', methods=['GET'])
@require_api_key
def get_phc_inventory(phc_name):
    phc_name = phc_name.replace('_', ' ')
    doc = phc_inventory_collection.find_one({'phcName': phc_name})
    if not doc:
        return jsonify({'phc': phc_name, 'items': []})
    return jsonify({'phc': phc_name, 'items': doc.get('items', [])})

@app.route('/inventory/phc/<phc_name>/alerts', methods=['GET'])
@require_api_key
def phc_inventory_alerts(phc_name):
    phc_name = phc_name.replace('_', ' ')
    doc = phc_inventory_collection.find_one({'phcName': phc_name})
    if not doc:
        return jsonify({'alerts': []})
    items = doc.get('items', [])
    today = datetime.now().date()
    expired = []
    low_stock = []
    for it in items:
        try:
            exp = datetime.strptime(it.get('expiry', '2099-01-01'), '%Y-%m-%d').date()
            if exp < today:
                expired.append({'name': it.get('name'), 'batch': it.get('batch', '-'), 'expiry': it.get('expiry')})
        except Exception:
            pass
        try:
            if int(it.get('stock', 0)) < LOW_STOCK_THRESHOLD:
                low_stock.append({'name': it.get('name'), 'stock': int(it.get('stock', 0))})
        except Exception:
            pass
    return jsonify({'expired': expired, 'low_stock': low_stock})

@app.route('/inventory/phc/<phc_name>', methods=['POST'])
@require_api_key
def update_phc_inventory(phc_name):
    """Replace PHC inventory doc. Body must contain items list."""
    payload = request.json or {}
    items = payload.get('items')
    if items is None:
        return jsonify({'error': 'items required'}), 400
    phc_name = phc_name.replace('_', ' ')
    phc_inventory_collection.update_one({'phcName': phc_name}, {'$set': {'items': items, 'updatedAt': datetime.utcnow().isoformat()}}, upsert=True)
    return jsonify({'status': 'ok'})

# Simple webhook placeholder to send alerts (user should replace with real implementation)
@app.route('/alerts/webhook', methods=['POST'])
@require_api_key
def webhook_alerts():
    payload = request.json or {}
    # Example: {'phc': 'Wagholi PHC', 'type': 'low_stock', 'items': [...]}
    logger.info("Webhook received: %s", json.dumps(payload)[:500])
    # TODO: integrate with email/sms provider or FCM
    return jsonify({'status': 'received'})

# Basic metrics / openapi
@app.route('/metrics', methods=['GET'])
@require_api_key
def metrics():
    # lightweight metrics
    total_orders = requests_collection.count_documents({}) if requests_collection else 0
    total_phcs = len(phc_inventory_collection.distinct('phcName')) if phc_inventory_collection else 0
    return jsonify({'total_orders': total_orders, 'total_phcs': total_phcs, 'model_loaded': os.path.exists(MODEL_PATH)})

@app.route('/openapi.json', methods=['GET'])
def openapi():
    spec = {
        'openapi': '3.0.0',
        'info': {'title': 'SwasthyaAI API', 'version': '1.0'},
        'paths': {
            '/predict-demand': {'get': {'summary': 'Predict', 'parameters': []}},
            '/train': {'post': {'summary': 'Train model'}},
        }
    }
    return jsonify(spec)

# Admin: allow export of recent orders CSV for offline inspection
@app.route('/export/orders.csv', methods=['GET'])
@require_api_key
def export_orders_csv():
    df = df_from_requests()
    if df.empty:
        return jsonify({'error': 'no_data'}), 400
    csv = df.to_csv(index=False)
    return app.response_class(csv, mimetype='text/csv')

# Small utility: summarize PHC performance
@app.route('/phc/compare', methods=['POST'])
@require_api_key
def compare_phcs():
    payload = request.json or {}
    phcs = payload.get('phcs', [])
    if not isinstance(phcs, list) or len(phcs) < 2:
        return jsonify({'error': 'provide at least two phcs in list'}), 400
    results = []
    for p in phcs:
        orders = list(requests_collection.find({'phc': {'$regex': p, '$options': 'i'}}))
        total = len(orders)
        delivered = len([o for o in orders if o.get('status') == 'Delivered'])
        results.append({'phc': p, 'totalOrders': total, 'delivered': delivered, 'fulfillmentRate': round((delivered/total*100) if total>0 else 0, 2)})
    return jsonify(results)

# ------------------------
# Run
# ------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    logger.info("Starting SwasthyaAI on port %d", port)
    app.run(host='0.0.0.0', port=port)
