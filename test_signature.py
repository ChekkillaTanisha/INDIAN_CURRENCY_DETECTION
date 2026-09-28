import json

from rsa_verify import verify_signature

with open("blockchain_log.json", "r") as f:
    chain = json.load(f)

last_block = chain[-1]

result = verify_signature(
    last_block["hash"],
    last_block["signature"]
)

print("Signature Valid:", result)