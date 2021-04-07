import time  # 引入time模块

ticks = time.time()
print("当前时间戳为:", ticks)

import time

localtime = time.asctime(time.localtime(time.time()))
localtime=time.localtime(time.time())
localtime.tm_mon+30
print("本地时间为 :", localtime)
import calendar
cal = calendar.month(2016, 1,2)
