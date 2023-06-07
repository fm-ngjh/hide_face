import cv2
import numpy as np
from deepface import DeepFace

class hideFace:
    def __init__(self):
        # 検出した顔が持つパラメータ
        self.params = [] # [x1, y1, x2, y2, score, fx, fy, emotion]
        
    def readImg(self, imgPath):
        self.imgPath = imgPath
        self.img = cv2.imread(imgPath, -1)

    def detectFace(self):
        #画像の中心座標
        self.cx = self.img.shape[1] / 2   
        self.cy = self.img.shape[0] / 2 
        
        # 顔の検出
        demography = DeepFace.analyze(self.img, actions = ['emotion'], detector_backend = "retinaface")
        
        for face in demography:
            # 顔を囲む長方形の頂点座標の計算
            x1 = face["region"]["x"]
            y1 = face["region"]["y"]
            x2 = x1 + face["region"]["w"]
            y2 = y1 + face["region"]["h"]
            emotion = face["dominant_emotion"]
            
            #顔の中心座標
            fx = (x1 + x2) / 2    
            fy = (y1 + y2) / 2
            
            # スコアの計算
            score = 100 * abs((x1 - x2) / self.img.shape[1]) * abs((y1 - y2) / self.img.shape[0])  
            
            # パラメータの保存
            tmp = [x1, y1, x2, y2, score, fx, fy, emotion]
            self.params.append(tmp)
    
    def sortParams(self):
        # paramsをscoreで降順にソートする
        for i in range(len(self.params)-1):
            for j in range(0, len(self.params)-i-1):
                if(self.params[j][4] < self.params[j+1][4]):
                    tmp = self.params[j]
                    self.params[j] = self.params[j+1]
                    self.params[j+1] = tmp
        self.maxScore = self.params[0][4]
    
    def mosaic(self, params):
        # 長方形範囲を抜き出して縮小したあと拡大→画質が悪くなってモザイクになる
        img = self.img[params[1] : params[3], params[0] : params[2]]
        img = cv2.resize(img, None, fx=0.1, fy=0.1)
        img = cv2.resize(img, [params[2]-params[0], params[3]-params[1]])
        self.img[params[1] : params[3], params[0] : params[2]] = img
    
    def hide_with_stamp_by_emotion(self, params):
        # emotionから使用するスタンプを決める
        if params[7] == "angry":
            stamp = cv2.imread("./stamps/angry.png", -1)
        elif params[7] == "disgust":
            stamp = cv2.imread("./stamps/disgust.png", -1)
        elif params[7] == "fear":
            stamp = cv2.imread("./stamps/fear.png", -1)
        elif params[7] == "happy":
            stamp = cv2.imread("./stamps/happy.png", -1)
        elif params[7] == "surprise":
            stamp = cv2.imread("./stamps/surprise.png", -1)
        elif params[7] == "sad":
            stamp = cv2.imread("./stamps/sad.png", -1)
        else:
            stamp = cv2.imread("./stamps/neutral.png", -1)
    
        # 顔を囲む長方形を正方形に直して切り出す
        w = params[2] - params[0]
        h = params[3] - params[1]
        x1, x2, y1, y2 = 0, 0, 0 ,0
        if w <= h:
            d = (h - w) / 2
            y1, y2, x1, x2 = params[1] + int(d), params[3] - int(d), params[0], params[2]
        else:
            d = (w - h) / 2
            y1, y2, x1, x2 = params[1], params[3], params[0] + int(d), params[2] - int(d)
            
        stamp = cv2.resize(stamp, [x2 - x1, y2 - y1])
        
        # スタンプを合成 スタンプは透過pngなのでスタンプのピクセルのアルファ値が正の時にピクセルを上書きする
        for i in range(y2 - y1):
            for j in range(x2 - x1):
                if stamp[i, j][3] != 0:
                    self.img[y1 + i, x1 + j][:3] = stamp[i, j][:3]      
            
    def drawRectangles(self):
        # 顔を長方形で囲む メインの被写体は赤線，そうでない場合は青線   テスト用
        for i in range(len(self.params)):
            print("No" + str(i+1) + " " + str(self.params[i][4]))
            data = "No" + str(i+1) + ":" + str('{:.2f}'.format(self.params[i][4]) + "%" + " " + self.params[i][7])
            
            if  self.params[i][4] > self.maxScore / 2:    # メインとそれ以外をスコアによって区別する
                cv2.rectangle(self.img, (self.params[i][0], self.params[i][1]), (self.params[i][2], self.params[i][3]), (0, 0, 255), 1)
                cv2.putText(self.img, data, (self.params[i][0], self.params[i][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            else:
                cv2.rectangle(self.img, (self.params[i][0], self.params[i][1]), (self.params[i][2], self.params[i][3]), (255, 0, 0), 1)
                cv2.putText(self.img, data, (self.params[i][0], self.params[i][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    
    def hide_face(self):
        # メイン被写体はスタンプ，それ以外はモザイクで顔を隠す
        for i in range(len(self.params)):
            print("No" + str(i+1) + " " + str(self.params[i][4]))           
            if  self.params[i][4] > self.maxScore / 2:    # メインとそれ以外をスコアによって区別する
                self.hide_with_stamp_by_emotion(self.params[i])
            else:
                self.mosaic(self.params[i])
    
    def showImg(self):
        cv2.imshow("window", self.img)
        cv2.setMouseCallback('window', self.click_pos)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def click_pos(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            img2 = np.copy(self.img)
            cv2.circle(img2,center=(x,y),radius=5,color=255,thickness=-1)
            pos_str='(x,y)=('+str(x)+','+str(y)+')'
            cv2.putText(img2,pos_str,(x+10, y+10),cv2.FONT_HERSHEY_PLAIN,2,255,2,cv2.LINE_AA)
            cv2.imshow('window', img2)
                    
    def hideFace_main(self, imgPath):
        self.readImg(imgPath)
        self.detectFace()
        self.sortParams()
        #self.drawRectangles()
        self.hide_face()
        self.showImg()
        
#imgPath = "./resource/holi.jpg"
imgPath = "./resource/children.jpg"

hf = hideFace()
hf.hideFace_main(imgPath)

