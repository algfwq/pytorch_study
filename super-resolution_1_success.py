import cv2
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

# input_location = 'http://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/results/output_test_pasd/0fbc3855c7cfdc95.png'
input_location = 'input.jpg'
prompt = ''
output_image_path = 'result.png'

os.environ['MODELSCOPE_CACHE'] = r'D:\pytorch_study\models'

input = {
    'image': input_location,
    'prompt': prompt,
    # 'upscale': 2,
    'upscale': 4,
    'fidelity_scale_fg': 1.0,
    'fidelity_scale_bg': 1.0
}

# pasd = pipeline(Tasks.image_super_resolution_pasd, model='damo/PASD_v2_image_super_resolutions')
pasd = pipeline(Tasks.image_super_resolution_pasd, model='damo/PASD_v2_image_super_resolutions',
                download_path=r'D:\pytorch_study\models', device_map="auto")
output = pasd(input)[OutputKeys.OUTPUT_IMG]
cv2.imwrite(output_image_path, output)
print('pipeline: the output image path is {}'.format(output_image_path))
