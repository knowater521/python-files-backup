#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:    pfwu
@E-mail:    wpf0610@mail.ustc.edu.cn
@file:      03_GetSignatures.py
@software:  pycharm
Created on  2018/5/17 0017 19:42

"""


from wxpy import *
import re
import pandas as pd
import wordcloud
from scipy.misc import imread
import numpy as np
import jieba

def get_friends():

    bot = Bot()
    my_friends = bot.friends()
    return my_friends


def write_to_txt(filename,data):
    with open(filename,'a') as f:
        f.write(data)


def save_items(friends):
    l = ['昵称', '备注', '性别', '城市', '签名']
    df = pd.DataFrame(index=range(len(friends)), columns=l)

    for i,friend in enumerate(friends):
        remark = friend.remark_name # 备注
        if remark=='':
            remark = '无备注'
        nick = friend.nick_name # 昵称
        province = friend.province
        city = friend.city
        sex = friend.sex # 1 男 2
        if sex == 1:
            sex = '男'
        elif sex == 2:
            sex = '女'
        else:
            sex = '未知'

        signature = friend.signature
        df['昵称'][i] = nick
        df['备注'][i] = remark
        df['性别'][i] = sex
        df['城市'][i] = province+city
        df['签名'][i] = signature
    df.to_csv('01.csv',encoding="utf_8_sig")

def get_signatures(friends):

    for friend in friends:
        signature = friend.signature
        pattern = re.compile(r'[一-]+')
        filterdata = re.findall(pattern,signature)

        write_to_txt('signatures.txt',''.join(filterdata))

def generate_wordcloud():
    with open('signatures.txt','r') as f:
        content = f.read()

    segment = jieba.lcut(content)
    words_df = pd.DataFrame({'segment':segment})
    words_df.to_csv('02.csv',encoding='utf_8_sig')
    stopwords = pd.read_csv('stopwords.txt',index_col=False,quoting=3,sep="",names=['stopword'],encoding='utf-8')
    words_df = words_df[~words_df.segment.isin(stopwords.stopword)]

    # 词频统计
    words_stat = words_df.groupby(by=['segment'])['segment'].agg({'计数':np.size})
    words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending=False)

    print('jj')



if __name__ == '__main__':

    friends = get_friends()
    # save_items(friends)
    get_signatures(friends)
    generate_wordcloud()
