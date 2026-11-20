"""Dataset package with lazy training-loader imports."""

_LOADER_NAMES = {
    "get_image_dataloader", "get_video_dataloader", "get_audio_dataloader", "get_t2va_dataloader"
}

def __getattr__(name):
    if name in _LOADER_NAMES:
        from . import loaders
        return getattr(loaders, name)
    raise AttributeError(name)
