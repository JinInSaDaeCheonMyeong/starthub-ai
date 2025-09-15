import os
import re
import requests
import uvicorn
from pinecone import Pinecone, PodSpec
from fastapi import FastAPI, Header, HTTPException
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

app = FastAPI(
    title="Starthub AI Recommendation API",
    description="사용자 관심사에 기반하여 채용 공고를 추천하는 API",
    version="1.1.0"
)

model = None
index = None
INDEX_NAME = "announcement"

def clean_html(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()

def fetch_jobs_from_server():
    base_url = "https://api.start-hub.kr/announcements"
    all_announcements = []
    try:
        initial_response = requests.get(base_url, params={"page": 0, "size": 20})
        initial_response.raise_for_status()
        initial_data = initial_response.json().get("data", {})
        total_pages = initial_data.get("totalPages", 1)
        print(f"총 {total_pages} 페이지의 공고가 있습니다.")

        for page_num in range(total_pages):
            print(f"{page_num + 1}/{total_pages} 페이지를 가져오는 중...")
            response = requests.get(base_url, params={"page": page_num, "size": 20})
            response.raise_for_status()
            announcements_on_page = response.json().get("data", {}).get("content", [])
            if announcements_on_page:
                all_announcements.extend(announcements_on_page)
    except requests.exceptions.RequestException as e:
        print(f"[오류] 공고 API 요청 중: {e}")
        return []
    print(f"총 {len(all_announcements)}개의 공고를 가져왔습니다.")
    return all_announcements

def update_jobs_in_pinecone():
    jobs = fetch_jobs_from_server()

    vectors_to_upsert = []
    for job in jobs:
        url = job.get('url', '')
        match = re.search(r'pbancSn=(\d+)', url)
        if not match:
            continue
        job_id = match.group(1)

        title = job.get('title', '')
        description = clean_html(job.get('content', ''))
        text_to_embed = f"{title}. {description}"
        
        vector = model.encode(text_to_embed, convert_to_tensor=False).tolist()
        metadata = {"title": title, "url": url, "description": description[:300] + "..."}
        vectors_to_upsert.append((job_id, vector, metadata))

    if vectors_to_upsert:
        for i in range(0, len(vectors_to_upsert), 100):
            batch = vectors_to_upsert[i:i+100]
            index.upsert(vectors=batch)
        print(f"Pinecone에 {len(vectors_to_upsert)}개의 공고 벡터 저장/업데이트 완료.")

@app.on_event("startup")
def startup_event():
    global model, index

    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    dimension = model.get_sentence_embedding_dimension()

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric='cosine',
            spec=PodSpec(environment=PINECONE_ENVIRONMENT)
        )
    index = pc.Index(INDEX_NAME)
    print(f"'{INDEX_NAME}' 인덱스에 연결되었습니다.")

    if index.describe_index_stats()['total_vector_count'] == 0:
        update_jobs_in_pinecone()

    scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    scheduler.add_job(update_jobs_in_pinecone, 'cron', hour=1, minute=0)
    scheduler.start()
    
    print("--- 서버 초기화 완료 ---")

@app.get("/recommend", summary="공고 추천 받기")
def get_recommendations(authorization: str = Header(None)):
    if model is None or index is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization 헤더에 유효한 Bearer 토큰이 필요합니다.")
    
    token = authorization.split("Bearer ")[1]

    try:
        user_interests_response = requests.get("https://api.start-hub.kr/user/me", headers={"Authorization": f"Bearer {token}"})
        user_interests_response.raise_for_status()
        user_interests = user_interests_response.json().get("data", {}).get("startupFields", [])
        if not user_interests:
            raise HTTPException(status_code=404, detail="사용자 관심사 정보를 찾을 수 없습니다.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"사용자 정보 API 요청 실패: {e}")

    query_text = ", ".join(user_interests)
    query_vector = model.encode([query_text], convert_to_tensor=False).tolist()
    response = index.query(vector=query_vector, top_k=3, include_metadata=True)
    
    recommendations = []
    for match in response['matches']:
        recommendations.append({
            "title": match['metadata']['title'],
            "url": match['metadata']['url'],
            "score": match['score']
        })

    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)