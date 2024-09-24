from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import torch
model_pose = YOLO(r'../Models/mypose5.pt').cuda()
model_people = YOLO(r'../Models/player_V8.pt').cuda()
model_ball = YOLO(r'../Models/ball_V8.pt').cuda()

#read video
video = "Clip/vbtA3"
cap = cv2.VideoCapture(video+".mp4")
_, img = cap.read()

def fillnan(l):
    for i in range(len(l)):
        if l[i] == None:
            l[i] = (0,0)
    return l

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(CORNER) < 4:
        print(x, ' ', y) 
        CORNER.append((x,y))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
    if len(CORNER) == 4:
        cv2.destroyAllWindows()
        
def Get_corner():
    global CORNER
    def sort(points):
        bottom_left = min(points, key=lambda point: (point[1], point[0]))
        top_right = max(points, key=lambda point: (point[1], point[0]))
        
        points.remove(bottom_left)
        points.remove(top_right)
    
        bottom_right = min(points, key=lambda point: point[0])
        top_left = max(points, key=lambda point: point[0])
    
        return [bottom_left, bottom_right, top_left, top_right]
    
    cv2.imshow('image', img) 
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    
    CORNER = sort(CORNER)
    return CORNER

#將存下的資料畫出
def draw_people(dst, list_poeple, num, list_id):
    id_colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 青色
        (128, 0, 0),    # 深红
        (0, 128, 0),    # 深绿
        (0, 0, 128),    # 深蓝
        (128, 128, 0),  # 橄榄色
        (128, 0, 128),  # 紫色
        (0, 128, 128),  # 深青
        (192, 192, 192),# 灰色
        (255, 165, 0),  # 橙色
        (128, 0, 128),  # 紫色
        (255, 192, 203),# 粉色
        (0, 128, 128),  # 深青
        (0, 0, 0),      # 黑色
        (255, 255, 128),# 浅黄色
        (128, 255, 128),# 浅绿色
        (255, 128, 128),# 浅红色
        (128, 128, 255),# 浅蓝色
        (255, 0, 128),  # 粉红色
        (128, 0, 255),  # 紫红色
        (255, 128, 0),  # 橘黄色
        (0, 255, 128),  # 浅绿色
        (128, 255, 255),# 淡蓝色
        (255, 128, 255),# 粉紫色
        (128, 255, 255),# 浅青色
        (192, 0, 192),  # 中性紫色
   ]
    before=[None]*40
    f=0
    for i in range(len(list_people)):
        for j in range(len(list_people[i])):
            colorr=id_colors[int(list_id[i][j])]
            x = list_people[i][j][0]
            y = list_people[i][j][1]
            cv2.circle(dst, (x, y), radius = 3, color = colorr, thickness = -1)
            if(f==0):
                before[int(list_id[i][j])]=(x,y)
                #cv2.circle(dst, (x, y), radius = 10, color = colorr, thickness = -1)
            else:
                if(before[int(list_id[i][j])]!=None):
                    cv2.line(dst, before[int(list_id[i][j])], (x, y), colorr,3) 
                #else: cv2.circle(dst, (x, y), radius = 10, color = colorr, thickness = -1)
                before[int(list_id[i][j])]=(x,y)
        f+=1
                
            
def draw_pose(dst, list_pose, pose_kind,Ser1,Ser2):
    for i in range(len(list_pose)):
        if(list_pose[i] == None):
            continue
        
        for j in range(len(list_pose[i])):
            x = list_pose[i][j][0]
            y = list_pose[i][j][1]
            if(pose_kind[i][j] == 0):
                #cv2.rectangle(dst,(x-5,y+80-5),(x+5,y+80+5),(130,130,130),3)
                cv2.circle(dst, (x+10, y+80), radius = 6, color = [130,130,130], thickness = -1)
                Ser1[i]=(x,y)
                print(Ser1[i])
            else: 
                Ser2[i]=(x,y)
                #cv2.circle(dst, (x+10 ,y+80), radius = 6, color = [188,188,188], thickness = -1)
                #cv2.rectangle(dst,(x-2,y-2),(x+2,y+2),(188,188,188),3)

def draw(list_people, list_pose, pose_kind, num, list_id,Ser1,Ser2):
    dst = np.ones([520, 960, 3], dtype = np.uint8)*255
    cv2.line(dst, (480, 80), (480, 440), (255,0,0), 5) #中線
    cv2.line(dst, (120, 80), (120, 440), (255,0,0), 5)
    cv2.line(dst, (120, 80), (840, 80), (255,0,0), 5)
    cv2.line(dst, (120, 440), (840, 440), (255,0,0), 5)
    cv2.line(dst, (840, 80), (840, 440), (255,0,0), 5)
    
    #draw_people(dst, list_people, num, list_id)
    draw_pose(dst, list_pose, pose_kind,Ser1,Ser2)
    
    #save
    #address=add+str(1)+'-'+str(4)+'.jpg'
    #cv2.imwrite(address, dst)
    '''
    cv2.imshow("Image", dst)
    cv2.waitKey(0)
    '''

CORNER = []
CORNER = Get_corner()

mapping = np.float32([[120, 80], [120, 440], [840, 80], [840, 440]])
transform = cv2.getPerspectiveTransform(np.float32(CORNER), mapping)

#初始化變數
FRAME = 0
count = 0
num = 0
list_people = []
list_pose = []
pose_kind = []
ball_list=[]
list_id = []
Ser1=[None]*90
Ser2=[None]*90
Ser3=[(None,None)]*90


#動作控制
set_happend = 0
spike_happend = 0
save_pic=0

while True:
    ret, frame = cap.read()

    if ret: img = frame
    else: break

    FRAME += 1
    print(FRAME)
    
    result_pose = model_pose(img) 
    result_people = model_people.track(img, tracker="bytetrack.yaml", persist=True)
    result_ball = model_ball(img)
    
    people = []
    pose = []
    kind = []
    idd = []
    ball=[]
    
   
    for i in range(len(result_ball[0].boxes)):
        box = result_ball[0].boxes[i]
        tensor = box.xyxy[0]
        x1 = int(tensor[0].item())
        y1 = int(tensor[1].item())
        x2 = int(tensor[2].item())
        y2 = int(tensor[3].item())
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        #在img框球
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,125,125),3)
        ball.append((cx,cy))
      
    print(result_ball)

    for i in range(len(result_people[0].boxes)):
        box = result_people[0].boxes[i]
        tensor = box.xyxy[0]
        x1 = int(tensor[0].item())
        y1 = int(tensor[1].item())
        x2 = int(tensor[2].item())
        y2 = int(tensor[3].item())
        label=int(box.id.item())
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        
        #人投影
        head_position = np.float32([[cx, y2]])
        transformed_head_position = cv2.perspectiveTransform(head_position.reshape(1,1,2), transform)[0]
        site=[int(transformed_head_position[0,0]),int(transformed_head_position[0,1])]
        people.append(site)
        idd.append(box.id)
        
        #在img框人
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.putText(img, str(label), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 5, cv2.LINE_AA)
        
    for i in range(len(result_pose[0].boxes)):
        box=result_pose[0].boxes[i]
        tensor = box.xyxy[0]
        x1 = int(tensor[0].item())
        y1 = int(tensor[1].item())
        x2 = int(tensor[2].item())
        y2 = int(tensor[3].item())
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        
        #動作投影
        head_position = np.float32([[cx, y2]])
        transformed_head_position = cv2.perspectiveTransform(head_position.reshape(1,1,2), transform)[0]
        site=[int(transformed_head_position[0,0]), int(transformed_head_position[0,1])]
        pose.append(site)
        
        #辨識model_pose抓到的動作類別 0:set 1:spike
        if(box.cls == 0):
            kind.append(0)
            set_happend = 1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)
        else:
            if(set_happend == 1):
                spike_happend = 1
            kind.append(1)
            cv2.rectangle(img,(x1,y1),(x2,y2),(125,125,0),3)
    
    #if set發生 存yolo抓到的位置
    if(set_happend == 1):
        count += 1
        list_people.append(people)
        list_pose.append(pose)
        pose_kind.append(kind)
        list_id.append(idd)
        ball_list.append(ball)
    
    #if count>150，將list、count、set_happend清空        
    if(count > 90 or (count > 60 and spike_happend == 0)):
        list_people = []
        list_pose = []
        pose_kind = []
        list_id = []
        ball_list=[]
        count = 0
        set_happend = 0
        spike_happend = 0
    
    #if count==150，將圖片存下來
    if(count == 90):
        num += 1
        draw(list_people, list_pose, pose_kind, num, list_id,Ser1,Ser2)
        save_pic=1
        break
    #缺:when count==149，將list資料投影並畫成跑位圖，存下來
    print(count)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
    
#可能影片結束時count不足150
if(count>0 and count < 90 and save_pic==0):
    num += 1
    draw(list_people, list_pose, pose_kind, num, list_id,Ser1,Ser2)
    
cap.release()
cv2.destroyAllWindows()

for i in range(len(ball_list)):
    if(ball_list[i]==None): continue
    for j in range(len(ball_list[i])):
        x=ball_list[i][j][0]
        y=ball_list[i][j][1]
        Ser3[i]=(x,y)

Ser1 = fillnan(Ser1)
Ser2 = fillnan(Ser2)
Ser3 = pd.DataFrame(Ser3,columns=["ball_x","ball_y"])
Ser3 = Ser3.interpolate(method="linear", axis=0)
Ser3 = Ser3.fillna(0)
Ser1 = pd.DataFrame(Ser1, columns = ["set_x", "set_y"])
Ser2 = pd.DataFrame(Ser2, columns = ["spike_x", "spike_y"])

data = pd.concat([Ser1,Ser2,Ser3], axis=1)
#d.to_csv('D:/vbtresult2/vbtA6'+".csv")
