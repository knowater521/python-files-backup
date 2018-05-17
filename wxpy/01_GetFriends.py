
from wxpy import *

# 初始化机器人，扫码登陆
#

bot = Bot()

# █
# Getting uuid of QR code.
# Downloading QR code.
# Please scan the QR code to log in.
# Please press confirm on your phone.
# Loading the contact, this may take a little while.
# Login successfully as XXXX

my_friends = bot.friends()
print(type(my_friends)) # <class 'wxpy.api.chats.chats.Chats'>