import cv2
from script.screen_test import grab_screen
window1 = (605,404,635,430)

while(True):

    screen_gray = cv2.cvtColor(grab_screen(window1),cv2.COLOR_BGR2GRAY)#灰度图像收集
    
    cv2.imshow('window1',screen_gray)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break