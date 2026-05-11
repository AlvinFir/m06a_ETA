import joblib
import pandas as pd
import polars as pl
import os
from datetime import datetime

class TrafficPredictor:
    def __init__(self, model_dir, data_dir):
        # 加載模型與轉換器
        self.eta_model = joblib.load(os.path.join(model_dir, "final_lgbm_model.joblib"))
        self.traffic_model = joblib.load(os.path.join(model_dir, "rf_traffic_model.pkl"))
        self.le = joblib.load(os.path.join(model_dir, "gantry_encoder.pkl"))
        
        # 讀取拓撲資料
        self.topo_df = pl.read_parquet(os.path.join(data_dir, "gantry_topology_final.parquet"))
        # 建立查詢字典加速檢索
        self.topo_lookup = {row['GantryID']: row for row in self.topo_df.to_dicts()}

    def _volume_to_speed_info(self, volume, free_flow_speed=110):
        """參考程式碼中的流量轉速度邏輯"""
        load_factor = volume / 4800
        if load_factor < 0.6:
            speed_idx = 1.0 - (load_factor * 0.05)
        elif load_factor < 0.9:
            speed_idx = 0.95 - (load_factor - 0.6) * 0.83
        else:
            speed_idx = max(0.4, 0.7 - (load_factor - 0.9) * 2.0)
        return speed_idx, free_flow_speed * speed_idx

    def predict_trip(self, start_id, end_id, departure_time, vehicle_type=31):
        # 1. 執行拓撲尋路 (鏈結接龍)
        path_nodes = []
        current_id = start_id
        max_hops = 150 
        found = False

        for _ in range(max_hops):
            node_data = self.topo_lookup.get(current_id)
            if not node_data:
                break
            path_nodes.append(node_data)
            if current_id == end_id:
                found = True
                break
            current_id = node_data['Next_GantryID']
            if current_id == "ENDPOINT" or current_id is None:
                break
        
        if not found:
            return {"error": f"路徑不連通或找不到 ID: {start_id} -> {end_id}"}

        # 2. 逐段預測
        total_sec = 0
        total_km = 0
        
        # 確保 departure_time 是 datetime 物件
        if isinstance(departure_time, str):
            # 處理前端傳來的 ISO 字串 (如 2026-05-07T09:42)
            departure_time = datetime.fromisoformat(departure_time.replace('Z', ''))

        for row in path_nodes:
            # --- [A. 流量預測 (RF)] ---
            g_code = self.le.transform([row['GantryID']])[0]
            traffic_X = pd.DataFrame([[
                g_code, departure_time.hour, departure_time.weekday(),
                row['緯度'], row['經度'], 1200, 0.15
            ]], columns=['Gantry_Code', 'hour', 'weekday', '緯度', '經度', 'Vol_Lag1', 'Heavy_Vehicle_Ratio'])

            pred_vol = self.traffic_model.predict(traffic_X)[0]

            # --- [B. 流量轉化] ---
            s_idx, s_avg = self._volume_to_speed_info(pred_vol)

            # --- [C. ETA 耗時預測 (LGBM)] ---
            # 關鍵修正：確保這裡有 7 個特徵，且名稱與訓練時一致
            eta_X = pd.DataFrame([{
                "VehicleType": vehicle_type,
                "hour": departure_time.hour,
                "weekday": departure_time.weekday(),
                "Segment_Length": row['Segment_Length'],
                "Gantry_Avg_Speed_Current": s_avg,
                "Heavy_Vehicle_Rate": 0.15, # 注意：名稱需與訓練資料一致
                "Speed_Index": s_idx
            }])

            sec = self.eta_model.predict(eta_X)[0]
            total_sec += sec
            total_km += row['Segment_Length']

        # 3. 結算
        return {
            "start_name": path_nodes[0]['起點交流道'],
            "end_name": path_nodes[-1]['迄點交流道'],
            "total_km": round(total_km, 2),
            "total_minutes": int(total_sec // 60),
            "total_seconds": int(total_sec % 60),
            "avg_speed": round(total_km / (total_sec / 3600), 1) if total_sec > 0 else 0,
            "estimated_toll": round(max(0, (total_km - 20.0) * 1.2), 0),
            "path_gantries": [n['GantryID'] for n in path_nodes]
        }