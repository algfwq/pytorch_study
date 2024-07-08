from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

os.environ['MODELSCOPE_CACHE'] = r'D:\pytorch_study\models'

video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/000.mp4'
video_super_resolution_pipeline = pipeline(
    Tasks.video_super_resolution,
    'damo/cv_msrresnet_video-super-resolution_lite', download_path=r'D:\pytorch_study\models',
    device_map="auto")
result = video_super_resolution_pipeline(video)[OutputKeys.OUTPUT_VIDEO]
