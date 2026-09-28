import hashlib
import time
import json

import os


def load_chain():
    if os.path.exists("blockchain_log.json"):
        with open("blockchain_log.json", "r") as f:
            return json.load(f)
    return []


chain = load_chain()


def add_record(data):

    chain = load_chain()

    timestamp = str(time.time())

    previous_hash = (
        chain[-1]["hash"]
        if len(chain) > 0
        else "0"
    )

    block_data = (
        str(data)
        + timestamp
        + previous_hash
    )

    block_hash = hashlib.sha256(
        block_data.encode()
    ).hexdigest()

    from rsa_signature import sign_data

    signature = sign_data(block_hash)

    block = {
        "timestamp": timestamp,
        "data": data,
        "previous_hash": previous_hash,
        "hash": block_hash,
        "signature": signature.hex()
    }

    print("ADD_RECORD CALLED")

    chain.append(block)

    with open("blockchain_log.json", "w") as f:
        json.dump(chain, f, indent=4)

    return block


def verify_chain():

    chain = load_chain()

    from rsa_signature import verify_signature

    # Empty blockchain
    if len(chain) == 0:
        return {
            "valid": False,
            "hash_check": False,
            "link_check": False,
            "rsa_check": False,
            "hash_issues": [],
            "link_issues": [],
            "rsa_issues": [],
            "reason": "Blockchain is empty."
        }

    # ============================================================
    # ISSUE LISTS
    # ============================================================

    hash_issues = []
    link_issues = []
    rsa_issues = []

    # ============================================================
    # 1. HASH INTEGRITY CHECK
    # ============================================================

    for i, block in enumerate(chain):

        block_number = i + 1

        try:

            recalculated_hash = hashlib.sha256(
                (
                    str(block["data"])
                    + block["timestamp"]
                    + block["previous_hash"]
                ).encode()
            ).hexdigest()

        except Exception:

            hash_issues.append(block_number)
            continue

        if recalculated_hash != block["hash"]:

            hash_issues.append(block_number)

    # ============================================================
    # 2. PREVIOUS HASH LINK CHECK
    # ============================================================

    for i, block in enumerate(chain):

        block_number = i + 1

        if i == 0:

            # First block must point to "0"
            if block["previous_hash"] != "0":

                link_issues.append(block_number)

        else:

            if (
                block["previous_hash"]
                != chain[i - 1]["hash"]
            ):

                link_issues.append(block_number)

    # ============================================================
    # 3. RSA SIGNATURE CHECK
    # ============================================================

    for i, block in enumerate(chain):

        block_number = i + 1

        try:

            signature = bytes.fromhex(
                block["signature"]
            )

        except Exception:

            rsa_issues.append(block_number)
            continue

        if not verify_signature(
            block["hash"],
            signature
        ):

            rsa_issues.append(block_number)

    # ============================================================
    # FINAL CHECK STATUS
    # ============================================================

    hash_check = len(hash_issues) == 0
    link_check = len(link_issues) == 0
    rsa_check = len(rsa_issues) == 0

    valid = (
        hash_check
        and link_check
        and rsa_check
    )

    # ============================================================
    # FINAL RESULT
    # ============================================================

    if valid:

        reason = "Blockchain Valid"

    else:

        reason = "Blockchain tampering detected."

    return {
        "valid": valid,
        "hash_check": hash_check,
        "link_check": link_check,
        "rsa_check": rsa_check,
        "hash_issues": hash_issues,
        "link_issues": link_issues,
        "rsa_issues": rsa_issues,
        "reason": reason
    }