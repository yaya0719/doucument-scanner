import cv2
import imutils
import transform_points
from skimage.filters import threshold_local

image = cv2.imread('11.jpg')
# 检查图像是否成功加载
if image is None:
    print("无法加载图像，请检查路径或文件是否存在。")
    exit()

# 计算比例并调整大小
orig = image.copy()
ratio=orig.shape[0]/500.0
image = imutils.resize(orig,height=500)

#轉灰階圖片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#圖片模糊，模糊後可去雜訊(圖片，kernal，標準差)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#找圖片邊緣(圖片，最低門檻值，最高門檻值)
edged = cv2.Canny(gray, 75, 200)

#test
#cv2.imshow("Outline", edged)
#cv2.waitKey(0)

#找輪廓(圖片，內外輪廓，壓縮的方法)
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#可能有多個物品所以保留最大前5個
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt=None

#找有四個輪廓的圖形
for c in cnts:
    peri = cv2.arcLength(c, True) #計算輪廓周長(輪廓，輪廓是否封閉)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # 如果轮廓有四个点，假设它是文档
    if len(approx) == 4:
        screenCnt = approx
        break

# 如果找到四边形轮廓
if screenCnt is not None:
    warped_image = transform_points.four_points_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
    # 将透视变换后的图像转换为灰度图像
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    
    # 使用局部阈值进行扫描效果
    T = threshold_local(warped_image, 11, offset=10, method="gaussian")
    warped = (warped_image > T).astype("uint8") * 255
    
    # 显示扫描效果
    cv2.imshow("Scanned", warped)
    cv2.waitKey(0)
    
    # 保存扫描效果的图像
    cv2.imwrite('scanned_output.png', warped)
else:
    print("没有找到文档轮廓。")

cv2.destroyAllWindows()
