import uuid
import hashlib
import code

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

pc = code.prpcrypt('keyskeyskeyskeys')  # 初始化密钥  # 初始化密钥
s_encrypt = pc.encrypt(str(license_str()))   # <class 'bytes'>

file_path='/Users/liyueyan/Desktop'
file_path = file_path + '/license_.lic'
s_encrypt = str(s_encrypt, encoding = "utf-8")   #  bytes to str
with open(file_path, 'w', encoding='utf-8') as lic:
    lic.write(str(s_encrypt))
    lic.close()




