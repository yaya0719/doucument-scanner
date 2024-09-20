import cv2
import imutils
import transform_points

image = cv2.imread('test4.jpg')

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
cv2.imshow("Outline", edged)
cv2.waitKey(0)

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

#如果有四邊輪廓
if screenCnt is not None:
    #取得透視變換後的照片
    warped_image = transform_points.four_points_transform(orig, screenCnt.reshape(4, 2) * ratio)
    #將照片轉換成灰色再處理
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    # 二質化，這裡使用自適應二值化會自動設定閾值
    # cv2.adaptiveThreshold(img, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    # img 來源影像
    # maxValue 最大灰度，通常設定 255
    # adaptiveMethod 自適應二值化計算方法
    # thresholdType 二值化轉換方式
    # blockSize 轉換區域大小，通常設定 11
    # C 偏移量，通常設定 2
    adaptive_thresh = cv2.adaptiveThreshold(gray_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 10)
    cv2.imshow("Scanned", adaptive_thresh)
    cv2.waitKey(0)
    cv2.imwrite('scanned_output.png', adaptive_thresh)
else:
    print("沒有找到四邊輪廓的文檔。")

cv2.destroyAllWindows()
