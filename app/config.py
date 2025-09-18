import os
import re
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

INDEX_NAME = "announcement"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
PINECONE_BATCH_SIZE = 100
MAX_METADATA_DESCRIPTION_LENGTH = 300
HTML_TAG_PATTERN = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
STARTHUB_API_URL = "https://api.start-hub.kr/announcements"
