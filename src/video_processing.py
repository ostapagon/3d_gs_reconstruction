import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
# import cv2
import os

# def read_video(file):
#     capture = cv2.VideoCapture(str(file))
#     fps = capture.get(cv2.CAP_PROP_FPS)
#     n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

#     frames = []

#     for _ in tqdm(np.arange(n_frames)):
#         success, image_cv = capture.read()

#         if not success:
#             break

#         frame = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
#         frames.append(frame)

#     return frames, fps, n_frames

# def video2images(video_path, output_dir_path=None, resize_scale=0.5):
#     frames, fps, n_frames = read_video(Path(video_path))

#     if output_dir_path is not None:
#         output_dir_path = Path(output_dir_path)
#         images_output_dir_path = output_dir_path/'input'
#     else:
#         output_dir_path = Path(str(Path(video_path)).split('.')[0])
#         images_output_dir_path = output_dir_path/"images"

#     images_output_dir_path.mkdir(parents=True, exist_ok=True)

#     for idx, img in enumerate(frames):
#         file_name = os.path.join(images_output_dir_path, f"image{idx}.jpg")
#         cv2.imwrite(file_name, cv2.cvtColor(cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale), cv2.COLOR_RGB2BGR))
    
#     return output_dir_path


if __name__ == "__main__":
    # print(video2images("/app/data/raw_video/room.mp4"))
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())