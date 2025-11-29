"""Script to fetch Ethereum ENS data"""

from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Optional
from dotenv import load_dotenv
import datetime
import gzip
import os
import json
import time

import requests

from environ.constant import ENS_ENDPOINT, DATA_PATH

load_dotenv()

DOMAINS = """
    id
    name
    labelName
    labelhash
    parent {
        id
    }
    resolver {
        id
        texts
        address
    }
    resolvedAddress {
        id
    }
    resolver{
        address
    }
    ttl
    isMigrated
    createdAt
    owner {
        id
    }
    registrant {
        id
    }
    wrappedOwner {
        id
    }
    expiryDate
    registration{
        id
    }
"""


def query_structurer(series: str, spec: str, arg: str = "") -> str:
    """Structure a GraphQL query."""

    # format query arguments
    if arg != "":
        arg = "(" + arg + ")"

    # format query content
    q = series + arg + "{" + spec + "}"
    return q


@retry(wait=wait_exponential(min=2, max=10), stop=stop_after_attempt(3))
def graphdata(
    *q, url: str = ENS_ENDPOINT, headers: Optional[dict[str, str]] = None
) -> dict:
    """Fetch data from a GraphQL endpoint."""

    # pack all subqueries into one big query concatenated with linebreak '\n'
    query = "{" + "\n".join(q) + "}"
    r = requests.post(url, json={"query": query}, headers=headers, timeout=60)

    response_json = json.loads(r.text)
    time.sleep(0.5)
    return response_json


def query(
    save_path: str,
    series: str,
    query_template: str,
    headers: Optional[dict[str, str]] = None,
    time_var: str = "created",
    end_point: str = ENS_ENDPOINT,
    batch_size: int = 1000,
):
    """Query data and save to a file."""

    # Interrupt and resume
    if os.path.exists(save_path):
        with gzip.open(save_path, "rt") as f:
            lines = f.readlines()
            if lines:
                # Get the last created timestamp and add 1 to avoid duplication
                last_created = str(int(json.loads(lines[-1])[time_var]) + 1)
            else:
                last_created = "0"
    else:
        last_created = "0"

    # Fetch data
    with gzip.open(save_path, "at") as f:

        while True:
            # Query data
            reservepara_query = query_structurer(
                series,
                query_template,
                arg=f'first: {batch_size}, orderBy: "{time_var}", '
                + f"orderDirection: asc, where: {{{time_var}_gte: {last_created}}}",
            )
            res = graphdata(reservepara_query, url=end_point, headers=headers)
            # Pagination check
            if "data" in set(res):
                if res["data"][series]:
                    # Process fetched rows
                    rows = res["data"][series]
                    length = len(rows)

                    # Update last_created timestamp
                    last_created = rows[-1][time_var]
                    print(
                        f"Fetched {datetime.datetime.fromtimestamp(int(last_created))}"
                    )

                    if length == batch_size:
                        # Remove the last_created update from the write operation
                        rows = [row for row in rows if row[time_var] != last_created]
                        # Write remaining rows to file
                        f.write("\n".join([json.dumps(row) for row in rows]) + "\n")
                    else:
                        # Write rows to file and break the loop
                        f.write("\n".join([json.dumps(row) for row in rows]) + "\n")
                        break
                else:
                    break
            else:
                raise ValueError("Error in fetching data")


if __name__ == "__main__":
    query(
        save_path=f"{DATA_PATH}/ens_domains.jsonl.gz",
        series="domains",
        query_template=DOMAINS,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('THEGRAPH_API_KEY')}",
        },
        time_var="createdAt",
    )
