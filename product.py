import os
import json

import requests
from dotenv import load_dotenv


load_dotenv()
USERNAME, PASSWORD = os.getenv("OXYLABS_USERNAME"), os.getenv("OXYLABS_PASSWORD")

payload = {
    "source": "amazon_product",
    "query": "B0BWHZJHPL",
    "parse": True,
    "domain": "com",
    "geo_location": "11249"
}

response = requests.request(
    "POST",
    "https://realtime.oxylabs.io/v1/queries",
    auth=(USERNAME, PASSWORD),
    json=payload,
)

print(response.json())

with open("product.json", "w") as f:
    json.dump(response.json()["results"], f, indent=4)