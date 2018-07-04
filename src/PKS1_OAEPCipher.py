from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA


class PKS1_OAEPCipher(object):
    def readEncryptionKey(self, file):
        encryptionKeyHandle = open(file, 'r')
        self.encryptionKey = RSA.importKey(encryptionKeyHandle.read())
        encryptionKeyHandle.close()

    def readDecryptionKey(self, file):
        decryptionKeyHandle = open(file, 'r')
        self.decryptionKey = RSA.importKey(decryptionKeyHandle.read())
        decryptionKeyHandle.close()

    def getEncryptionKey(self):
        return self.encryptionKey

    def getDecryptionKey(self):
        return self.decryptionKey

    def encrypt(self, raw):
        cipher = PKCS1_OAEP.new(self.encryptionKey)
        ciphertext = cipher.encrypt(raw)
        return ciphertext

    def decrypt(self, enc):
        cipher = PKCS1_OAEP.new(self.decryptionKey)
        message = cipher.decrypt(enc)
        return message
