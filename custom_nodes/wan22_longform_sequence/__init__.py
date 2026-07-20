from __future__ import annotations

import os

import folder_paths


class Wan22ConditionalSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "wan22_longform/segment"}),
                "format": (["auto", "mp4"], {"default": "mp4"}),
                "codec": (["auto", "h264"], {"default": "h264"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "save_video"
    CATEGORY = "Wan22 Longform/sequence"
    OUTPUT_NODE = True

    def save_video(self, enabled, video, filename_prefix, format, codec):
        if not enabled:
            return (video,)

        from comfy_api.latest._util import VideoContainer

        width, height = video.get_dimensions()
        output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        extension = VideoContainer.get_extension(format)
        output_path = os.path.join(output_folder, f"{filename}_{counter:05}_.{extension}")
        video.save_to(output_path, format=VideoContainer(format), codec=codec, metadata=None)
        return (video,)


class Wan22TailFramesFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tail_frames": ("INT", {"default": 8, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tail_frames",)
    FUNCTION = "select_tail"
    CATEGORY = "Wan22 Longform/sequence"

    def select_tail(self, images, tail_frames):
        return (images[-tail_frames:],)


NODE_CLASS_MAPPINGS = {
    "Wan22ConditionalSaveVideo": Wan22ConditionalSaveVideo,
    "Wan22TailFramesFromBatch": Wan22TailFramesFromBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22ConditionalSaveVideo": "Wan22 Conditional Save Video",
    "Wan22TailFramesFromBatch": "Wan22 Tail Frames from Batch",
}
