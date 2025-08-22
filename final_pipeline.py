import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from llm import answer_with_tone

from dotenv import load_dotenv
load_dotenv()  # this loads .env variables into os.environ
from embed import EmbeddingStore
import re
from clean import clean_query


if __name__ == "__main__":
    store = EmbeddingStore()


    # query
    query = input("Enter your query: ")
    while query != "exit":
        filtered_query = clean_query(query)
        chunks = store.query(filtered_query, top_k=3)
        print(chunks)
        reply = answer_with_tone(
            query=query,
            chunks=[r["text"] for r in chunks],
            tone="concise, friendly",
            temperature=0.6,
            max_tokens=1024,
            stream=False,
        )
        print(reply)
        query = input("Enter your query: ")

