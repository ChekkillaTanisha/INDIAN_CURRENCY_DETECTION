import json
from datetime import datetime


def get_note_history(serial_number):

    try:

        with open(
            "blockchain_log.json",
            "r"
        ) as f:

            chain = json.load(f)

    except:

        return {
            "times_scanned": 0,
            "first_seen": "N/A",
            "last_seen": "N/A"
        }

    matches = []

    for block in chain:

        data = block["data"]

        if (
            data.get("serial_number")
            == serial_number
        ):

            matches.append(block)

    if len(matches) == 0:

        return {
            "times_scanned": 0,
            "first_seen": "N/A",
            "last_seen": "N/A"
        }

    first_timestamp = float(
        matches[0]["timestamp"]
    )

    last_timestamp = float(
        matches[-1]["timestamp"]
    )

    return {

        "times_scanned": len(matches),

        "first_seen": datetime.fromtimestamp(
            first_timestamp
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        ),

        "last_seen": datetime.fromtimestamp(
            last_timestamp
        ).strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    }