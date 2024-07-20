import pymongo
import dotenv
import os
import requests

dotenv.load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MONGO_DB_URI = os.getenv("MONGO_DB_CONNECTION_STRING")


client = pymongo.MongoClient(MONGO_DB_URI)
db = client.sample_mflix
collection = db.movies

embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
        json={"inputs": text},
    )

    if response.status_code != 200:
        raise ValueError(
            f"Request failed with status code {response.status_code}: {response.text}"
        )

    return response.json()


query = "imaginary characters from outer space at war"

results = collection.aggregate(
    [
        {
            "$vectorSearch": {
                "queryVector": generate_embedding(query),
                "path": "plot_embedding_hf",
                "numCandidates": 100,
                "limit": 4,
                "index": "PlotSemantivSearch",
            }
        }
    ]
)

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')
