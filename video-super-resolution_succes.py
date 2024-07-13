from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import shutil

os.environ['MODELSCOPE_CACHE'] = r'D:\pytorch_study\models'

video = r'D:\pytorch_study\input\aa.mp4'
output_video_path = r'D:\pytorch_study\output\super_resolved_video.mp4'  # 指定输出视频的路径

video_super_resolution_pipeline = pipeline(
    Tasks.video_super_resolution,
    'damo/cv_msrresnet_video-super-resolution_lite', download_path=r'D:\pytorch_study\models',
    device_map="auto")

# 运行超分辨率模型
result = video_super_resolution_pipeline(video)[OutputKeys.OUTPUT_VIDEO]

# 检查 result 是否为视频路径
if isinstance(result, str):
    # 将处理后的视频复制到指定路径
    shutil.copy(result, output_video_path)
    print(f"视频已保存到: {output_video_path}")
else:
    print("处理结果不是有效的视频路径")
