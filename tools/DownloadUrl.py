import urllib.request as request
import time
import os
import socket
from shutil import copyfile

#下载测试图片
def saveItems(path,fileP):
    socket.setdefaulttimeout(60)
    if not path:
        path = '/bigdata/style/test_photo/'
    if not fileP:
        fileP = '/bigdata/style/style-recog-samples/test.txt'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print(path,fileP)
    with open(fileP) as lines:
        for line in lines:
            arr = line.strip().replace('\n','').split(',')
            name = arr[0]
            fileName= path + '/' + name + '.jpg'
            url = arr[1]
            exists = os.path.exists(fileName)
            print(fileName,exists)
            retry = 1
            while retry<5:
                try:
                    if not exists:
                        request.urlretrieve(url,fileName)
                    retry = 6
                except BaseException as e:
                    print('download img error.')
                    print(e)
                    retry += 1
                    exists = False
                time.sleep(1)
        lines.close()

saveItems(None,None)
#save test data
#saveItems('/bigdata/style/train_photo/','/bigdata/style/style-recog-samples/train+val.txt')