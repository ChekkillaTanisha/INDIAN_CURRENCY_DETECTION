import json
import hashlib

with open("blockchain_log.json", "r") as f:
    chain = json.load(f)

block = chain[1]   # Block #2

block_data = (
    str(block["data"])
    + block["timestamp"]
    + block["previous_hash"]
)

new_hash = hashlib.sha256(
    block_data.encode()
).hexdigest()

print("New Hash:")
print(new_hash)