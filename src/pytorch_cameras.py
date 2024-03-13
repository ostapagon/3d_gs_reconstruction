import torch
import numpy as np

import torch.nn.functional as F
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import *


default_camera_kwargs = {
    'znear': 0.01,
    'zfar': 100,
}


def get_lookat_transform(ro=(1.,1,1), lookat=(0,0,0.), world2cam=True):
    ro = torch.tensor(ro, dtype=torch.float32).unsqueeze(1)
    lookat = torch.tensor(lookat, dtype=torch.float32).unsqueeze(1)

    f = F.normalize(lookat - ro,dim=0)
    r = torch.cross(torch.tensor([0,1.,0]).unsqueeze(1), f)
    u = torch.cross(f, r)

    I = torch.tensor([
        [1,0.,0],
        [0,1,0],
        [0, 0,1]
    ])
    
    R = torch.cat([r,u,f], dim=1).T
    T = ro.squeeze()

    R = I @ R

    if world2cam:
        cam2world = torch.cat([torch.cat([R, T[None]], dim=0), torch.tensor([0,0,0,1])[:, None]], dim=1)
        
        world2cam = torch.linalg.inv(cam2world)
        R = world2cam[:3,:3]
        T = world2cam[-1, :3]
    
    
    return R, T

def write_video(frames_np, out_video_path, fps=25, audiofile=None):
    clip = ImageSequenceClip(frames_np, fps=fps)
    if audiofile:
        audioclip = AudioFileClip(audiofile).set_duration(clip.duration)
        new_audioclip = CompositeAudioClip([audioclip])
        clip = clip.set_audio(new_audioclip)
    clip.write_videofile(str(out_video_path), fps=fps)


def create_camera(R, T):
    return FoVPerspectiveCameras(R=R[None], T=T[None], **default_camera_kwargs)

def circle_orbit(t):
    return np.array([np.cos(t), 0, np.sin(t)])


def point2camera(point, LOOK_AT_HEAD):
    R, T = get_lookat_transform(point, LOOK_AT_HEAD)
    camera = create_camera(R=R, T=T)
    return camera

def generate_orbit(LOOK_AT_HEAD = (0,0,0.), CAMERA_BASE = (0, 0, 0.), N_FRAMES = 300, R = 3.):    
    LOOK_AT_HEAD = np.array(LOOK_AT_HEAD)
    LOOK_AT_HEAD = np.array(CAMERA_BASE)

    orbit_points = np.array([CAMERA_BASE+R*circle_orbit(t) * .5*(np.sin(3*t)*.5 + 2) + np.array([0,0.01*np.sin(3*t),0]) for t in np.linspace(0, 2*np.pi, N_FRAMES)])
    orbit_cameras = [point2camera(p, LOOK_AT_HEAD) for p in orbit_points]
    return orbit_points, orbit_cameras
