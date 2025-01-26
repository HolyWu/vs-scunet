from __future__ import annotations

import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .__main__ import download_model
from .network_scunet import SCUNet

__version__ = "2.0.0"

os.environ["CI_BUILD"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "Additional .* warnings suppressed about get_attr references")
warnings.filterwarnings("ignore", "Attempted to insert a get_attr Node with no underlying reference")
warnings.filterwarnings("ignore", ".* does not reference an nn.Module")
warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class Backend:
    @dataclass
    class Torch:
        module: torch.nn.Module

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


class SCUNetModel(IntEnum):
    scunet_color_15 = 0
    scunet_color_25 = 1
    scunet_color_50 = 2
    scunet_color_real_psnr = 3
    scunet_color_real_gan = 4


@contextmanager
def redirect_stdout_to_stderr():
    old_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)


@redirect_stdout_to_stderr()
@torch.inference_mode()
def scunet(
    clip: vs.VideoNode,
    device_index: int = 0,
    num_streams: int = 1,
    batch_size: int = 1,
    model: SCUNetModel = SCUNetModel.scunet_color_real_gan,
    auto_download: bool = False,
    tile: list[int] = [0, 0],
    tile_pad: int = 8,
    trt: bool = False,
    trt_debug: bool = False,
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Practical Blind Denoising via Swin-Conv-UNet and Data Synthesis

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param batch_size:              Number of frames per batch.
    :param model:                   Model to use.
    :param auto_download:           Automatically download the specified model if the file has not been downloaded.
    :param tile:                    Tile width and height. As too large images result in the out of GPU memory issue, so
                                    this tile option will first crop input images into tiles, and then process each of
                                    them. Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("scunet: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("scunet: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("scunet: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("scunet: num_streams must be at least 1")

    if batch_size < 1:
        raise vs.Error("scunet: batch_size must be at least 1")

    if model not in SCUNetModel:
        raise vs.Error("scunet: model must be one of the members in SCUNetModel")

    if not isinstance(tile, list) or len(tile) != 2:
        raise vs.Error("scunet: tile must be a list with 2 items")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    inf_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    f2t_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    t2f_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]

    inf_stream_locks = [Lock() for _ in range(num_streams)]
    f2t_stream_locks = [Lock() for _ in range(num_streams)]
    t2f_stream_locks = [Lock() for _ in range(num_streams)]

    model_name = f"{SCUNetModel(model).name}.pth"

    if os.path.getsize(os.path.join(model_dir, model_name)) == 0:
        if auto_download:
            download_model(f"https://github.com/HolyWu/vs-scunet/releases/download/model/{model_name}")
        else:
            raise vs.Error(
                "scunet: model file has not been downloaded. run `python -m vsscunet` to download all models, or set "
                "`auto_download=True` to only download the specified model"
            )

    if all(t > 0 for t in tile):
        pad_w = math.ceil(min(tile[0] + 2 * tile_pad, clip.width) / 64) * 64
        pad_h = math.ceil(min(tile[1] + 2 * tile_pad, clip.height) / 64) * 64
    else:
        pad_w = math.ceil(clip.width / 64) * 64
        pad_h = math.ceil(clip.height / 64) * 64

    if trt:
        import tensorrt
        import torch_tensorrt
        from torch_tensorrt.dynamo.lowering._decomposition_groups import TORCH_TRT_DECOMPOSITIONS

        if torch.ops.aten.select_scatter.default in TORCH_TRT_DECOMPOSITIONS:
            del TORCH_TRT_DECOMPOSITIONS[torch.ops.aten.select_scatter.default]

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_batch-{batch_size}"
                + f"_{pad_w}x{pad_h}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            module = init_module(model_name, device, dtype)
            inputs = (torch.zeros([batch_size, 3, pad_h, pad_w], dtype=dtype, device=device),)
            exported_program = torch.export.export(module, inputs, strict=False)

            module = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs,
                device=device,
                enabled_precisions={dtype},
                debug=trt_debug,
                num_avg_timing_iters=4,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
            )

            torch_tensorrt.save(module, trt_engine_path, output_format="torchscript", inputs=inputs)

        module = [torch.jit.load(trt_engine_path).eval() for _ in range(num_streams)]
        backend = Backend.TensorRT(module)
    else:
        module = init_module(model_name, device, dtype)
        backend = Backend.Torch(module)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: vs.VideoFrame | list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with f2t_stream_locks[local_index], torch.cuda.stream(f2t_streams[local_index]):
            img = torch.stack([frame_to_tensor(f[i] if batch_size > 1 else f, device) for i in range(batch_size)])

            f2t_streams[local_index].synchronize()

        with inf_stream_locks[local_index], torch.cuda.stream(inf_streams[local_index]):
            if all(t > 0 for t in tile):
                output = tile_process(img, tile, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                if need_pad := pad_w - w > 0 or pad_h - h > 0:
                    img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "replicate")

                if trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                if need_pad:
                    output = output[:, :, :h, :w]

            inf_streams[local_index].synchronize()

        with t2f_stream_locks[local_index], torch.cuda.stream(t2f_streams[local_index]):
            frame = tensor_to_frame(output[0], f[0].copy() if batch_size > 1 else f.copy(), t2f_streams[local_index])
            for i in range(1, batch_size):
                frame.props[f"vsscunet_batch_frame{i}"] = tensor_to_frame(
                    output[i], f[0].copy() if batch_size > 1 else f.copy(), t2f_streams[local_index]
                )
            return frame

    if (pad := (batch_size - clip.num_frames % batch_size) % batch_size) > 0:
        clip = clip.std.DuplicateFrames([clip.num_frames - 1] * pad)

    clips = [clip[i::batch_size] for i in range(batch_size)]

    outputs = [clips[0].std.FrameEval(lambda n: clips[0].std.ModifyFrame(clips, inference), clip_src=clips)]
    for i in range(1, batch_size):
        outputs.append(outputs[0].std.PropToClip(f"vsscunet_batch_frame{i}"))

    output = vs.core.std.Interleave(outputs)
    if pad > 0:
        output = output[:-pad]
    return output


def init_module(model_name: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    module = SCUNet(config=[4, 4, 4, 4, 4, 4, 4])
    module.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location="cpu", mmap=True))
    return module.eval().to(device, dtype)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream) -> vs.VideoFrame:
    tensor = tensor.detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    stream.synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame


def tile_process(
    img: torch.Tensor,
    tile: list[int],
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Torch | Backend.TensorRT,
    index: int,
) -> torch.Tensor:
    height, width = img.shape[2:]

    # start with black image
    output = torch.zeros_like(img)

    tiles_x = math.ceil(width / tile[0])
    tiles_y = math.ceil(height / tile[1])

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile[0]
            ofs_y = y * tile[1]

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile[0], width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile[1], height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            h, w = input_tile.shape[2:]
            if need_pad := pad_w - w > 0 or pad_h - h > 0:
                input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), "replicate")

            # process tile
            if isinstance(backend, Backend.TensorRT):
                output_tile = backend.module[index](input_tile)
            else:
                output_tile = backend.module(input_tile)

            if need_pad:
                output_tile = output_tile[:, :, :h, :w]

            # output tile area on total image
            output_start_x = input_start_x
            output_end_x = input_end_x
            output_start_y = input_start_y
            output_end_y = input_end_y

            # output tile area without padding
            output_start_x_tile = input_start_x - input_start_x_pad
            output_end_x_tile = output_start_x_tile + input_tile_width
            output_start_y_tile = input_start_y - input_start_y_pad
            output_end_y_tile = output_start_y_tile + input_tile_height

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
