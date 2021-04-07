from Crypto.Cipher import AES
import base64
password = '1234567890123456' #秘钥
text = '1234567890123456' #需要加密的内容
text=text.encode("utf-8")
password=password.encode("utf-8")
model = AES.MODE_ECB #定义模式
aes = AES.new(password,model) #创建一个aes对象

en_text = aes.encrypt(text) #加密明文
print(en_text)
en_text = base64.encodebytes(en_text) #将返回的字节型数据转进行base64编码
print(en_text)
en_text = en_text.decode('utf8') #将字节型数据转换成python中的字符串类型
en_text=en_text.encode("utf-8")
print(en_text.strip())
# t='aaa'
# t=t.encode()
# ma = memoryview(t)
# aes.decrypt(ma,en_text)


