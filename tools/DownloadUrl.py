import urllib.request as request
import time
import os
import socket

#下载测试图片
def saveItems(path,fileP):
    socket.setdefaulttimeout(60)
    if not path:
        path = '../test_data/'
    if not fileP:
        print('please input the URLS file path.....')
        return
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print(path,fileP)
    with open(fileP) as lines:
        for line in lines:
            url = line.strip().replace('\n','')
            start = url.rindex('/') + 1
            fileName= path + url[start:]
            exists = os.path.exists(fileName)
            print('fileName:'+fileName,' success:'+ str(not exists))
            retry = 1
            while retry<3:
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