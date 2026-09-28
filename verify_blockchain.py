# verify_blockchain.py

import json
import hashlib
from scripts.rsa_signature import verify_signature

with open("blockchain_log.json", "r") as f:
    chain = json.load(f)

valid = True

for block in chain:

    block_data = (
        str(block["data"])
        + block["timestamp"]
        + block["previous_hash"]
    )

    calculated_hash = hashlib.sha256(
        block_data.encode()
    ).hexdigest()

    if calculated_hash != block["hash"]:
        print("HASH TAMPERED")
        valid = False
        continue

    signature_valid = verify_signature(
        block["hash"],
        bytes.fromhex(block["signature"])
    )

    if not signature_valid:
        print("SIGNATURE INVALID")
        valid = False

if valid:
    print("BLOCKCHAIN VERIFIED")