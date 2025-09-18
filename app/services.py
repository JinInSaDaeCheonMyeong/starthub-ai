import re
import requests
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer

from app import config

embedding_model = None
pinecone_index = None


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

        metadata = {"title": title, "url": url, "description": truncated_description}
        vectors_to_upsert.append((job_id, vector, metadata))

    if vectors_to_upsert:
        for i in range(0, len(vectors_to_upsert), config.PINECONE_BATCH_SIZE):
            batch = vectors_to_upsert[i:i + config.PINECONE_BATCH_SIZE]
            pinecone_index.upsert(vectors=batch)


def initialize_services():
    global embedding_model, pinecone_index

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
