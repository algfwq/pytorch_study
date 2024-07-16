import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

os.environ['MODELSCOPE_CACHE'] = r'D:\pytorch_study\models'

sr = pipeline(Tasks.image_super_resolution, model='damo/cv_rrdb_image-super-resolution', download_path=r'D:\pytorch_study\models',
              device_map="auto")
result = sr('input.jpg')
cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
