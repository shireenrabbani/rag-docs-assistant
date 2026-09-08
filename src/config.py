import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
GENERATION_MODEL = "claude-sonnet-5"

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5
