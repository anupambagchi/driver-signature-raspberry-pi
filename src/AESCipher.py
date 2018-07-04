from Crypto import Random
from Crypto.Cipher import AES

class AESCipher(object):
    def __init__(self):
        #self.key = hashlib.sha256(key.encode()).digest()
        self.key = Random.get_random_bytes(32)

    def setKey(self, key):
        self.key = key

    def getKey(self):
        return self.key

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        #return base64.b64encode(iv + cipher.encrypt(raw))
        return iv + cipher.encrypt(raw)

    def decrypt(self, enc):
        # enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (AES.block_size - len(s) % AES.block_size) * chr(AES.block_size - len(s) % AES.block_size)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]