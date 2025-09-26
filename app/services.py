import re
import requests
from datetime import datetime
from pinecone import Pinecone, PodSpec
import openai
import json
from sentence_transformers import SentenceTransformer
import numpy as np

from app import config

embedding_model = None
pinecone_index = None
openai_client = None


def clean_html(raw_html: str) -> str:
    clean_text = re.sub(config.HTML_TAG_PATTERN, '', raw_html)
    return clean_text.strip()


def fetch_jobs_from_server() -> list:
    all_announcements = []
    try:
        initial_response = requests.get(config.STARTHUB_API_URL, params={"page": 0, "size": 20})
        initial_response.raise_for_status()
        initial_data = initial_response.json().get("data", {})
        total_pages = initial_data.get("totalPages", 1)

        for page_num in range(total_pages):
            response = requests.get(config.STARTHUB_API_URL, params={"page": page_num, "size": 20})
            response.raise_for_status()
            announcements_on_page = response.json().get("data", {}).get("content", [])
            if announcements_on_page:
                all_announcements.extend(announcements_on_page)

    except requests.exceptions.RequestException as _:
        return []

    return all_announcements


def update_jobs_in_pinecone():
    global embedding_model, pinecone_index
    if not embedding_model or not pinecone_index:
        return

    jobs = fetch_jobs_from_server()
    if not jobs:
        return

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

        vector = embedding_model.encode(text_to_embed, convert_to_tensor=False).tolist()

        truncated_description = description[:config.MAX_METADATA_DESCRIPTION_LENGTH]
        if len(description) > config.MAX_METADATA_DESCRIPTION_LENGTH:
            truncated_description += "..."

        metadata = {
            "title": title,
            "url": url,
            "description": truncated_description,
            "organization": job.get('organization', ''),
            "region": job.get('region', ''),
            "startupHistory": job.get('startupHistory', ''),
            "supportField": job.get('supportField', ''),
            "receptionPeriod": job.get('receptionPeriod', ''),
            "is_active": True
        }
        vectors_to_upsert.append((job_id, vector, metadata))

    if vectors_to_upsert:
        for i in range(0, len(vectors_to_upsert), config.PINECONE_BATCH_SIZE):
            batch = vectors_to_upsert[i:i + config.PINECONE_BATCH_SIZE]
            pinecone_index.upsert(vectors=batch)


def initialize_services():
    global embedding_model, pinecone_index, openai_client

    embedding_model = SentenceTransformer(config.MODEL_NAME)
    embedding_dimension = embedding_model.get_sentence_embedding_dimension()

    pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)

    if config.INDEX_NAME not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=config.INDEX_NAME,
            dimension=embedding_dimension,
            metric='cosine',
            spec=PodSpec(environment=config.PINECONE_ENVIRONMENT)
        )
    pinecone_index = pinecone_client.Index(config.INDEX_NAME)

    if config.OPENAI_API_KEY:
        openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)


def get_recommendations(user_interests: list[str], liked_announcement_ids: list[str] = None) -> list:
    if not embedding_model or not pinecone_index:
        return []

    interest_query = ", ".join(user_interests)
    interest_vector = embedding_model.encode(interest_query, convert_to_tensor=False)

    final_vector = interest_vector

    if liked_announcement_ids:
        try:
            liked_vectors_response = pinecone_index.fetch(ids=liked_announcement_ids)
            liked_vectors = [vec.values for vec in liked_vectors_response.vectors.values()]

            if liked_vectors:
                average_liked_vector = np.mean(liked_vectors, axis=0)
                final_vector = 0.5 * interest_vector + 0.5 * average_liked_vector

        except Exception as e:
            print(f"'좋아요' 공고 벡터 조회 중 오류 발생: {e}")

    query_vector = final_vector.tolist()

    try:
        results = pinecone_index.query(
            vector=query_vector,
            top_k=20,
            include_metadata=True,
            filter={"is_active": {"$eq": True}}
        )
        return results.get('matches', [])
    except Exception as e:
        print(f"추천 검색 중 오류 발생: {e}")
        return []

def deactivate_expired_jobs():
    if not pinecone_index:
        return

    print("일일 작업 실행: 만료된 공고 비활성화 중...")
    try:
        stats = pinecone_index.describe_index_stats()
        total_vectors = stats['total_vector_count']
        if total_vectors == 0:
            return

        zero_vector = [0.0] * embedding_model.get_sentence_embedding_dimension()
        response = pinecone_index.query(
            vector=zero_vector,
            filter={"is_active": {"$eq": True}},
            top_k=10000,
            include_metadata=True
        )

        deactivated_count = 0
        for match in response.get('matches', []):
            reception_period = match.get('metadata', {}).get('receptionPeriod', '')
            if not reception_period:
                continue

            try:
                end_date_str = reception_period.split('~')[1].strip()
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M')
                if datetime.now() > end_date:
                    pinecone_index.update(id=match['id'], set_metadata={"is_active": False})
                    deactivated_count += 1
            except (IndexError, ValueError):
                continue
        
        print(f"{deactivated_count}개의 공고를 비활성화했습니다.")

    except Exception as e:
        print(f"비활성화 작업 중 오류 발생: {e}")

def search_jobs_by_natural_language(query: str, top_k: int = 10) -> list:
    if not all([embedding_model, pinecone_index, openai_client]):
        return []

    def _parse_query_with_openai(raw_query: str) -> dict:
        system_prompt = '''
        당신은 대한민국 채용 정보 게시판의 검색어 구문 분석 어시스턴트입니다.
        사용자의 원본 검색어가 주어지면, 이를 "semantic_query", "filters", "top_k" 세 개의 필드를 가진 구조화된 JSON 객체로 분해하세요.

        1. `semantic_query`: 시맨틱 벡터 검색을 위한 검색어의 핵심 부분입니다. 필터와 관련된 모든 키워드를 제거하세요. 이 값은 한국어여야 합니다.
        2. `filters`: 필터 객체의 목록입니다. 각 객체는 "field"와 "value"를 가져야 합니다.
           - 알려진 필터 `field` 이름은 `region`, `startupHistory` 입니다.
           - `region`의 경우, 일반적인 한국의 시/도 이름을 전체 공식 명칭으로 매핑하세요 (예: "서울" -> "서울특별시", "대구" -> "대구광역시", "경남" -> "경상남도").
        3. `top_k`: 요청된 결과의 수입니다 (예: "5개" -> 5). 지정되지 않은 경우 기본값은 10입니다.

        한국어 오타는 자연스럽게 처리하세요. 단어가 필터 키워드처럼 보이지만 철자가 틀린 경우, 이를 수정하세요.

        --- 예시 ---
        사용자 검색어: "서울에서 하는 예비창업자 IT 공고 3개"
        당신의 JSON 출력:
        {
          "semantic_query": "IT 공고",
          "filters": [
            {"field": "region", "value": "서울특별시"},
            {"field": "startupHistory", "value": "예비창업자"}
          ],
          "top_k": 3
        }
        ---

        출력은 다른 텍스트 없이 JSON 객체만이어야 합니다.
        '''
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_query}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    structured_query = _parse_query_with_openai(query)

    final_top_k = structured_query.get("top_k", top_k)
    semantic_query_text = structured_query.get("semantic_query", "")

    final_filter = {"is_active": {"$eq": True}}
    filters_from_llm = structured_query.get("filters", [])
    if filters_from_llm:
        pinecone_filters = [{f["field"]: {"$eq": f["value"]}} for f in filters_from_llm]
        final_filter["$and"] = pinecone_filters

    if not semantic_query_text and filters_from_llm:
        semantic_query_text = "창업 지원 공고"

    if not semantic_query_text:
        return []

    original_query_vector = embedding_model.encode(query, convert_to_tensor=False)
    semantic_query_vector = embedding_model.encode(semantic_query_text, convert_to_tensor=False)

    combined_vector = (0.7 * original_query_vector + 0.3 * semantic_query_vector)
    query_vector = combined_vector.tolist()

    try:
        results = pinecone_index.query(
            vector=query_vector,
            top_k=final_top_k,
            include_metadata=True,
            filter=final_filter
        )
        matches = results.get('matches', [])

        partial_matches = []
        other_matches = []
        query_stripped = query.strip()

        for match in matches:
            title = match.get('metadata', {}).get('title', '').strip()
            if query_stripped in title:
                partial_matches.append(match)
            else:
                other_matches.append(match)

        return partial_matches + other_matches

    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return []
