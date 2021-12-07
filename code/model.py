
import cv2
import numpy as np


class PlateDetector(object):
    ''' Define a plate detector.
    '''
    def __init__(self):
        ''' 初始化系列常量
        '''
        super(PlateDetector, self).__init__()

        self.test_img_dir = 'data/test/'    # directory to store test images
        self.temp_img_dir = 'data/temp/'    # directory to store temp images
        self.result_img_dir = 'data/result/'    # directory to store result images

        self.max_width = 500    # maxmium width of image
        self.min_area = 1000    # minimum area of a contour
        self.min_scale = 2    # 候选车牌区域宽/高比的最小值
        self.max_scale = 4    # 候选车牌区域宽/高比的最大值
        self.min_color_ratio = 0.3    # 颜色占比最小值


    def _resize(self, img):
        ''' 缩放过大的图片
        '''
        height, width = img.shape[0:2]
        
        if width > self.max_width:
            new_width = self.max_width
            new_height = int((self.max_width / width) * height)
            img = cv2.resize(img, (new_width, new_height))
        
        return img


    def _stretch(self, img):
        ''' 进行灰度变换
        '''
        min_val = img.min()
        max_val = img.max()

        img = (img - min_val) * (255 / (max_val - min_val))
        img = img.astype(np.uint8)

        return img 


    def _color_cnt(self, img, color):
        ''' 给定HSV图像，统计蓝、绿、黄三种颜色中某一种颜色像素数目占比
        '''
        if color == 'blue':
            # 蓝色范围
            lower = np.array([100, 43, 46])
            upper = np.array([124, 255, 255])
        elif color == 'yellow':
            # 黄色范围
            lower = np.array([26, 43, 46])
            upper = np.array([34, 255, 255])
        elif color == 'green':
            # 绿色范围
            lower = np.array([35, 43, 46])
            upper = np.array([77, 255, 255])
        else:
            raise Exception('Wrong at param "color" of function "_color_cnt" !')

        # 根据阈值进行计算
        mask = cv2.inRange(img, lower, upper)
        cnt = np.count_nonzero(mask)
        
        return cnt / (img.shape[0] * img.shape[1])


    def _locate(self, edge_img, orig_img):
        ''' 根据edge_img，对车牌区域进行定位
        '''
        # 获取轮廓集合
        # Note that 'contours' is a tuple of 'contour' and 'contour' is a ndarray 
        contours, hierachy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 根据轮廓面积初步筛选
        contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        # 获得轮廓的外接矩形
        rect_list = []
        for contour in contours:
            rect = cv2.boundingRect(contour)    # rect is a tuple, that is, (x, y, w, h)
            rect_list.append(rect)
    
        # 根据宽/高比进一步筛选
        rect_list = [r for r in rect_list if r[2] / r[3] >= self.min_scale and r[2] / r[3] <= self.max_scale]

        # 使用颜色占比识别判断出像车牌的区域
        results = []
        for rect in rect_list:
            # 截取原图中的区域，并转为HSV图像
            x, y, w, h = rect
            min_x, min_y, max_x, max_y = x, y, x + w, y + h
            block = orig_img[min_y:max_y, min_x:max_x]
            hsv_block = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
            
            value = self._color_cnt(hsv_block, 'blue')

            if value > self.min_color_ratio:
                results.append(rect)

        return results


    def solve(self, img_name):
        ''' Detect a plate in the image.
        '''
        img_path = self.test_img_dir + img_name
        img = cv2.imread(img_path)

        # check
        if img is None:
            raise Exception('Wrong! "img" is empty. Please check img_name!')

        # Resize
        img = self._resize(img)
        cv2.imwrite(filename=self.temp_img_dir + 'resize_' + img_name, img=img)
        # cv2.imshow('resize_img', img)
        # cv2.waitKey()

        # 高斯模糊
        gaussian_img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imwrite(filename=self.temp_img_dir + 'gaussian_' + img_name, img=gaussian_img)
        # cv2.imshow('gaussian_img', gaussian_img)
        # cv2.waitKey()
        
        # RGB图像转为灰度图像
        gray_img = cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(filename=self.temp_img_dir + 'gray_' + img_name, img=gray_img)
        # cv2.imshow('gray_img', gray_img)
        # cv2.waitKey()
    
        # 灰度拉伸
        stretch_img = self._stretch(gray_img)
        cv2.imwrite(filename=self.temp_img_dir + 'stretch_' + img_name, img=stretch_img)
        # cv2.imshow('stretch_img', stretch_img)
        # cv2.waitKey()

        # 开运算，去除噪声
        kernel = np.ones(shape=(20, 20), dtype=np.uint8)
        open_img = cv2.morphologyEx(stretch_img, cv2.MORPH_OPEN, kernel)
        
        # 加权重, 即 gray_img - open_img
        weight_img = cv2.addWeighted(gray_img, 1, open_img, -1, 0)
        cv2.imwrite(filename=self.temp_img_dir + 'weight_' + img_name, img=weight_img) 
        # cv2.imshow('weight_img', weight_img)
        # cv2.waitKey()

        # 图像二值化
        retval, binary_img = cv2.threshold(weight_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        cv2.imwrite(filename=self.temp_img_dir + 'binary_' + img_name, img=binary_img)
        # cv2.imshow('binary_img', binary_img)
        # cv2.waitKey()

        # Canny算子进行边缘检测
        edge_img = cv2.Canny(binary_img, 100, 200)    
        cv2.imwrite(filename=self.temp_img_dir + 'init_edge_' + img_name, img=edge_img)
        # cv2.imshow('init_edge_img', edge_img)
        # cv2.waitKey()

        # 消除小区域，连通大区域
        kernel = np.ones((10, 19), np.uint8)    
        edge_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)    # 闭运算
        edge_img = cv2.morphologyEx(edge_img, cv2.MORPH_OPEN, kernel)    # 开运算
        cv2.imwrite(filename=self.temp_img_dir + 'cross_edge_' + img_name, img=edge_img)
        # cv2.imshow('cross_edge_img', edge_img)
        # cv2.waitKey()

        # 定位车牌位置
        rect_list = self._locate(edge_img=edge_img, orig_img=img)
        
        # 绘制位置框
        for rect in rect_list:
            x, y, w, h = rect
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.imwrite(filename=self.result_img_dir + 'result_' + img_name, img=img)
        cv2.imshow('result_img', img)
        cv2.waitKey()
        
        cv2.destroyAllWindows()