from wxpy import *


bot = Bot()
my_friends = bot.friends()


sex_dict = {'男':0,'女':0}


for friend in my_friends:
    if friend.sex ==1:
        sex_dict['男'] += 1
    elif friend.sex == 2:
        sex_dict['女'] += 1

print(sex_dict)

chatting_friends = bot.chats(update=False)
print(chatting_friends)

