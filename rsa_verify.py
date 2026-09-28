from scripts.rsa_signature import public_key
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

def verify_signature(block_hash, signature):

    try:
        public_key.verify(
            bytes.fromhex(signature),
            block_hash.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        return True

    except:
        return False