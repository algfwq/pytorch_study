import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import os

os.environ['MODELSCOPE_CACHE'] = r'D:\pytorch_study\models'

# content_img = 'input.jpg'
content_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/dogs.jpg'
style_img = 'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/image-style-transfer/style_transfer_style.jpg'

style_transfer = pipeline(Tasks.image_style_transfer, model_id='damo/cv_aams_style-transfer_damo', download_path=r'D:\pytorch_study\models',
                          device_map="auto")
result = style_transfer(dict(content=content_img, style=style_img))
cv2.imwrite('result_style.png', result[OutputKeys.OUTPUT_IMG])
