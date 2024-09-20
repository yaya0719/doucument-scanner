import cv2
import numpy as np
#排序:左上 右上 右下 左下
def order_points(points):
    rect_points=np.zeros((4,2),dtype="float32") #四個點(x0,y0)到(x3,y3)
    #取得四個角落點的x和y之和，axis=1表示做水平方向相加
    xy_sum=points.sum(axis=1)
    #左上角和一定最小，右下角一定最大，np.argmin(xy_sum)會回傳xy_sum最小的index
    rect_points[0]=points[np.argmin(xy_sum)]
    rect_points[2]=points[np.argmax(xy_sum)]
    #取得四個角落點的y-x，axis=1表示做水平方向相減
    xy_diff=np.diff(points, axis=1)
    #右上角差一定最小，左下角一定最大
    rect_points[1]=points[np.argmin(xy_diff)]
    rect_points[3]=points[np.argmax(xy_diff)]
    return rect_points
#取得原照片轉換後的照片
def four_points_transform(image,points):
    rect_points=order_points(points)
    (tl,tr,br,bl)=rect_points #top button right left
    #尋找最大的寬，可能是top或button，因為兩點之間不一定是水平的，所以用畢式定理
    widthA=np.sqrt((br[0]-bl[0])**2+(br[1]-bl[1])**2)
    widthB=np.sqrt((tr[0]-tl[0])**2+(tr[1]-tl[1])**2)
    maxWidth=max(int(widthA),int(widthB))
     #尋找最大的高，可能是top或button，因為兩點之間不一定是水平的，所以用畢式定理
    heightA=np.sqrt((br[0]-tr[0])**2+(br[1]-tr[1])**2)
    heightB=np.sqrt((bl[0]-tl[0])**2+(bl[1]-tl[1])**2)
    maxHeight=max(int(heightA),int(heightB))
    #把四個角落放到整個視窗
    dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")
    #透視變換
    #取得要從rect_points變成dst的透視變換矩陣
    transform_matrix = cv2.getPerspectiveTransform(rect_points, dst)
    #用透視變換矩陣取得轉換後的照片
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))
    return warped
    
