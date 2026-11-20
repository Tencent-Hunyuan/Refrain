# This script contains two functions to create diffusion and transport objects.
#
# For the `create_diffusion` function, it is modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
#
# For the `create_transport` function, it is modified from the SiT repo:
#     SiT:   https://github.com/willisma/SiT/blob/main/transport/__init__.py
#
import torch
from .flow.transport import *

def create_transport(
        *,
        path_type,
        prediction,
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        snr_type="uniform",
        image_shift=1.0,
        audio_shift=None,
        video_shift=None,
        t2va_shift=None,
        reverse=False,
):
    """
    Create a Transport object for the flow matching schedulers.

    Args:
        *: A barrier to prevent positional arguments.
        path_type (str): Type of path to use. Can be "linear", "gvp", or "vp".
        prediction (str): Model prediction type. Can be "velocity", "score", or "noise".
        loss_weight (str, optional): Loss weight type. Can be "velocity", "likelihood", or None.
        train_eps (float, optional): Small epsilon for avoiding instability during training.
        sample_eps (float, optional): Small epsilon for avoiding instability during sampling.
        snr_type (str): Type of SNR to use. Can be "lognorm" or "uniform".
        shift (float): SNR shift value. It equals to sqrt(m/n) which is formulated by SD3.
        reverse (bool): Whether to reverse the flow.
        use_flux_shift (bool): Whether to use flux shift.
        flux_base_shift (float): Base shift value for flux.
        flux_max_shift (float): Max shift value for flux.
        flux_base_token (float): Base shift token for flux.
        flux_max_token (float): Max shift token for flux.
        flux_shift_factor (float): shift factor for flux.

    Returns:
        state (Transport): A Transport object for the flow matching schedulers.
    """
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    if snr_type == "lognorm":
        snr_type = SNRType.LOGNORM
    elif snr_type == "uniform":
        snr_type = SNRType.UNIFORM
    elif snr_type == "mix":
        snr_type = SNRType.MIX
    else:
        raise ValueError(f"Invalid snr type {snr_type}")

    if video_shift is None:
        video_shift = image_shift
    
    if audio_shift is None:
        audio_shift = 1.0
        
    path_choice = {
        "linear": PathType.LINEAR,
        "gvp": PathType.GVP,
        "vp": PathType.VP,
    }

    path_type = path_choice[path_type.lower()]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        snr_type=snr_type,
        image_shift=image_shift,
        video_shift=video_shift,
        audio_shift=audio_shift,
        t2va_shift=t2va_shift,
        reverse=reverse,
    )

    return state


def load_denoiser(args):
    if args.denoise_type == "flow":
        denoiser = create_transport(path_type=args.flow_path_type,
                                    prediction=args.flow_predict_type,
                                    loss_weight=args.flow_loss_weight,
                                    train_eps=args.flow_train_eps,
                                    sample_eps=args.flow_sample_eps,
                                    snr_type=args.flow_snr_type,
                                    image_shift=args.flow_shift_image,
                                    video_shift=args.flow_shift_video,
                                    audio_shift=getattr(args, 'flow_shift_audio', None),
                                    t2va_shift=getattr(args, 't2va_shift', None),
                                    reverse=args.flow_reverse,
                                    )
    else:
        raise ValueError(f"Unknown denoise type: {args.denoise_type}")
    return denoiser
