import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.predictor import TrafficPredictor

app = FastAPI(title="台灣國道預測系統")

# 允許跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化預測器
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
predictor = TrafficPredictor(
    model_dir=os.path.join(BASE_DIR, "models"),
    data_dir=os.path.join(BASE_DIR, "data")
)

# 資料模型
class PredictionRequest(BaseModel):
    start_id: str
    end_id: str
    departure_time: datetime
    vehicle_type: int = 31

@app.post("/predict")
async def predict(req: PredictionRequest):
    result = predictor.predict_trip(req.start_id, req.end_id, req.departure_time, req.vehicle_type)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # 補充座標點資訊供前端畫線
    path_details = []
    for gid in result["path_gantries"]:
        node = predictor.topo_lookup.get(gid)
        if node:
            path_details.append({"id": gid, "lat": node["緯度"], "lng": node["經度"]})
    result["path_details"] = path_details
    return result

@app.get("/gantries")
async def get_gantries():
    # 回傳所有門架 ID 與對應的起訖交流道名稱
    return predictor.topo_df.select(["GantryID", "起點交流道", "迄點交流道"]).to_dicts()

# 掛載靜態網頁 (確保你有 static 資料夾)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main.py:app", host="127.0.0.1", port=8000, reload=True)