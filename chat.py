# proactive chatting ability for her

import numpy as np
import time


def receive():
    return input()


if __name__ == '__main__':
    me = 'ウシオ'
    you = 'あなた'
    print("おはようございます！")
    time.sleep(1)
    print('今日はいい天気だか？')
    time.sleep(1)
    print('%sはこれからよろしくね！' % me)
    time.sleep(1)
    print('失礼ですけと、%sの名前は？' % you)
    s = receive()
    print('じゃ、これから%sと呼んで喜んでしょうか？' % s)
    s = receive()
    if s.startswith('はい') or s.startswith('いい'):
        print('ご')
