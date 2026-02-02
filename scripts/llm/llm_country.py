"""Script to label country"""

import numpy as np
import json

from environ.constant import PROCESSED_DATA_PATH
from environ.llm import ChatGPT
from tqdm import tqdm


COUNTRY_PROMPT = (
    'Please classify the country "{country}" '
    "into one of the following regions: Africa and Middle East, "
    "Asia and Pacific, Europe, Latin America and Caribbean, or "
    "North America. Your output should follow this format: "
    '{{"result": "<region>"}}.'
)

JSON_SCHEMA = {
    "name": "wallet",
    "schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
            },
        },
        "required": ["result"],
        "additionalProperties": False,
    },
    "strict": True,
}

gpt = ChatGPT(model="gpt-4o-mini")


# with open(PROCESSED_DATA_PATH / "anchor_country.json", "r", encoding="utf-8") as f:
#     anchor_country = json.load(f)

anchor_country = np.load(
    PROCESSED_DATA_PATH / "anchor" / "country_asc_with_china_5.npy",
    allow_pickle=True,
).tolist()

anchor_country_region = {}
for country in tqdm(anchor_country):
    response = gpt(
        COUNTRY_PROMPT.format(country=country),
        json_schema=JSON_SCHEMA,
    )
    anchor_country_region[country] = json.loads(response)["result"]

with open(
    PROCESSED_DATA_PATH / "anchor_country_region.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(anchor_country_region, f, indent=4)
