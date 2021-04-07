from Crypto.Cipher import AES
import base64
import uuid
import hashlib

class Encrypt:
    def __init__(self, key, iv):
        self.key = key.encode('utf-8')
        self.iv = iv.encode('utf-8')

    # @staticmethod
    def pkcs7padding(self, text):
        """
        明文使用PKCS7填充
        """
        bs = 16
        length = len(text)
        bytes_length = len(text.encode('utf-8'))
        padding_size = length if (bytes_length == length) else bytes_length
        padding = bs - padding_size % bs
        padding_text = chr(padding) * padding
        self.coding = chr(padding)
        return text + padding_text

    def aes_encrypt(self, content):
        """
        AES加密
        """
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        # 处理明文
        content_padding = self.pkcs7padding(content)
        # 加密
        encrypt_bytes = cipher.encrypt(content_padding.encode('utf-8'))
        # 重新编码
        result = str(base64.b64encode(encrypt_bytes), encoding='utf-8')
        return result

    def aes_decrypt(self, content):
        """
        AES解密
        """
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        content = base64.b64decode(content)
        text = cipher.decrypt(content).decode('utf-8'
                                              )
        return text.rstrip(self.coding)

def hash_msg(msg):
    sha256 = hashlib.sha256()
    sha256.update(msg.encode('utf-8'))
    res = sha256.hexdigest()
    return res

def get_mac_address():
    mac = uuid.UUID(int = uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0,11,2)])

mac_addr=get_mac_address()
active_date='10d'

#1、得到密钥，通过hash算法计算目标计算机的mac地址
psw = hash_msg('first_key' + str(mac_addr))
#2、新建一个license_str 的字典，用于保存真实的mac地址，license失效时间，加密后的字符串
license_str = {}
license_str['mac'] = mac_addr
license_str['time_str'] = active_date
license_str['psw'] = psw
# <class 'bytes'>


iv  = '0123928nv2i5ss69'
key = '63f09k56nv2b10cf'
a = Encrypt(key=key, iv=iv)
e=a.aes_encrypt(str(license_str))
#e = a.aes_encrypt('{"code":200,"data":{"apts":[]},"message":"","success":true}')
d = a.aes_decrypt(e)
print("加密:", e)
print("解密:", d)

file_path='/Users/liyueyan/Desktop'
file_path = file_path + '/license_.lic'
#s_encrypt = str(e, encoding = "utf-8")   #  bytes to str
with open(file_path, 'w', encoding='utf-8') as lic:
    lic.write(e)
    lic.close()