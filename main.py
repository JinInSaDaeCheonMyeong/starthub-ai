import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler

from app import models, services


@asynccontextmanager
async def lifespan(_: FastAPI):
    services.initialize_services()

    if services.pinecone_index.describe_index_stats()['total_vector_count'] == 0:
        services.update_jobs_in_pinecone()

    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(services.update_jobs_in_pinecone, 'cron', hour=1, minute=0)
    scheduler.start()
    yield


app = FastAPI(
    title="Starthub AI Recommendation API",
    description="사용자 관심사에 기반하여 채용 공고를 추천하는 API",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/recommend", summary="공고 추천 받기")
def get_recommendations(user_data: models.UserInterests):
    if services.embedding_model is None or services.pinecone_index is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")

    user_interests = user_data.interests
    if not user_interests:
        raise HTTPException(status_code=400, detail="관심사 목록이 비어있습니다.")

    query_text = ", ".join(user_interests)
    query_vector = services.embedding_model.encode([query_text], convert_to_tensor=False).tolist()

    query_response = services.pinecone_index.query(vector=query_vector, top_k=20, include_metadata=True)
    recommendations = []
    for result in query_response['matches']:
        recommendations.append({
            "title": result['metadata']['title'],
            "url": result['metadata']['url'],
            "score": result['score']
        })

    return {"recommendations": recommendations}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
