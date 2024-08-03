# -*- coding: utf-8 -*-
"""
Created on Wed May 1 09:56:10 2024

@author: FeiMa
Email:mafei0603@163.com
"""
import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import random


def get_video_comments(video_id, page=1):
    url = f'https://api.bilibili.com/x/v2/reply?jsonp=jsonp&pn={page}&type=1&oid={video_id}&sort=2'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    data = json.loads(response.text)
    comments = []
    for item in data['data']['replies']:
        comments.append({
            'user': item['member']['uname'],
            'content': item['content']['message']
        })
    return comments

if __name__ == '__main__':
    # 打开文件，'r' 表示读取模式  
    csv_filename = 'commentsShuangxiu.txt' #保存评论记录
    videoIDFile='videoID.txt'#保存待挖掘的视频文件ID
    finishedVideID='finishedVideID.txt'#已完成的ID记录
    with open(videoIDFile, 'r', encoding='utf-8') as file:  
    # 逐行读取文件  
        for line in file:  
        # 这里你可以对每一行进行处理，比如打印出来  
            video_id =  line[0:-1]
            with open(finishedVideID, 'a', encoding='utf-8', newline='') as finishfile:  
                csv_writer = csv.writer(finishfile)                        
                csv_writer.writerow(str(video_id))
                print('Input into finished file',video_id)
            page = 1
            while True:
                comments = get_video_comments(video_id, page)
                if not comments:
                    break
                for comment in comments:
                    print(f" {comment['user']},{comment['content']}")                   
                     
                    # 打开 CSV 文件进行写入
                    with open(csv_filename, mode='a', encoding='utf-8', newline='') as csvfile:
                        # 创建 CSV 写入器
                        csv_writer = csv.writer(csvfile)                        
                        csv_writer.writerow([comment])
                page += 1
                secRND = random.random()
                print(video_id,10+round(secRND*10),time.strftime(" Time: %Y-%m-%d %X",time.localtime()))
                time.sleep(13+round(secRND*10)) #延时随机时长，以免被检测到位爬虫
                 
