# proactive chatting ability for her

import numpy as np
import time


def receive():
    return input()


if __name__ == '__main__':
    me = 'ウシオ'
    you = 'あなた'
    print("おはようございます！")
    time.sleep(3)
    print('今日はいい天気だか？')
    time.sleep(3)
    print('%sはこれからよろしくね！' % me)
    time.sleep(3)
    print('失礼ですけと、%sの名前は？' % you)
    while True:
        s = receive()
        print('じゃ、これから%sと呼んで喜んでしょうか？' % s)
        s = receive()
        if s == 'はい' or s =='いい':
            print('さて、これで自己紹介は終わりです！一緒に楽しみにお話してくださいね！')
            break
        else:
            print('そうか、ごめん！本当のお名前は教えてくださいね！')
    print('終わり。。。')
    
