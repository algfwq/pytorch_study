import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

os.environ['MODELSCOPE_CACHE'] = r'D:\pytorch_study\models'

sr = pipeline('image-super-resolution-x2', model='bubbliiiing/cv_rrdb_image-super-resolution_x2', download_path=r'D:\pytorch_study\models',
              device_map="auto")
# result = sr(r'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg')
result = sr(r'D:\pytorch_study\input.jpg')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
