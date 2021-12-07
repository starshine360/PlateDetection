from code.model import PlateDetector



if __name__ == '__main__':
    myDetector = PlateDetector()
    img_name = 'test2.JPG'
    myDetector.solve(img_name)
