import os
import sys
from utils.embedding_loader import load_and_embed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


load_and_embed(
    csv_path="data/social/enriched_posts.json",
    persist_directory="db/social",
)