import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler

from app import models, services


@asynccontextmanager
async def lifespan(_: FastAPI):
    services.initialize_services()

    if services.pinecone_index.describe_index_stats()['total_vector_count'] == 0:
        print("인덱스가 비어있습니다. 초기 데이터 색인을 수행합니다...")
        services.update_jobs_in_pinecone()
        print("초기 데이터 색인이 완료되었습니다.")

    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(services.update_jobs_in_pinecone, 'cron', hour=0, minute=5)
    scheduler.add_job(services.deactivate_expired_jobs, 'interval', hours=1)
    scheduler.start()
    yield


app = FastAPI(
    title="Starthub AI API",
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

    results = services.get_recommendations(user_interests)
    recommendations = []
    for result in results:
        recommendations.append({
            "title": result['metadata']['title'],
            "url": result['metadata']['url'],
            "score": result['score']
        })

    return {"recommendations": recommendations}


@app.post("/search", summary="자연어 공고 검색")
def search_jobs(search_data: models.SearchQuery):
    if services.embedding_model is None or services.pinecone_index is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")

    if not search_data.query:
        raise HTTPException(status_code=400, detail="검색어가 비어있습니다.")

    results = services.search_jobs_by_natural_language(search_data.query)
    search_results = []
    for result in results:
        search_results.append({
            "title": result['metadata']['title'],
            "url": result['metadata']['url'],
            "score": result['score'],
            "metadata": {
                "organization": result['metadata'].get('organization', ''),
                "region": result['metadata'].get('region', ''),
                "startupHistory": result['metadata'].get('startupHistory', '')
            }
        })

    return {"results": search_results}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)