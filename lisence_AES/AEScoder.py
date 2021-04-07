import Crypto
from Crypto.Cipher import AES
import base64
#第一个问题解决方法：到自己的\Lib\site-packages目录找到Crypto文件，把这个文件开头改成大写
#第二个问题解决方法：pip3 install pycryptodome

class AEScoder():
    def __init__(self):
        self.__encryptKey = "iEpSxImA0vpMUAabsjJWug=="
        self.__key = base64.b64decode(self.__encryptKey)

    # AES加密
    def encrypt(self, data):
        BS = 16
        pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
        cipher = AES.new(self.__key, AES.MODE_ECB)
        encrData = cipher.encrypt(pad(data))
        # encrData = base64.b64encode(encrData)
        return encrData

    # AES解密
    def decrypt(self, encrData):
        # encrData = base64.b64decode(encrData)
        # unpad = lambda s: s[0:-s[len(s)-1]]
        unpad = lambda s: s[0:-s[-1]]
        cipher = AES.new(self.__key, AES.MODE_ECB)
        decrData = unpad(cipher.decrypt(encrData))
        return decrData.decode('utf-8')