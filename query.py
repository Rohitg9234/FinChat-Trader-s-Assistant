from embed import EmbeddingStore
import re
from clean import clean_query


if __name__ == "__main__":
    store = EmbeddingStore()


    # query
    query = "Even after identifying the correct trend direction on the chart, I often enter too early or too late, which causes me to miss profits or take unnecessary losses. How do I improve my entry timing?"
    query = clean_query(query)
    hits = store.query(query, top_k=1)
    for h in hits:
        print(h)