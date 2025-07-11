from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict
import uvicorn

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,  # 인증정보(쿠키 등)포함 가능
    allow_methods=["*"],  # 모든 HTTP 메서드(GET, POST 등) 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 라우터 등록
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])

# FAST 실행명령어 자동 실행
if __name__ == "__main__":
    uvicorn.run(
        app="main:app", host="0.0.0.0", port=8000, reload=True
    )  # reload=True : 코드 변경 시 서버 자동 재시작
