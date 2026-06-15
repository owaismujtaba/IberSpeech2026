import braindecode.models as bmodels

def create_model(model_name, n_classes, in_chans, input_window_samples, chs_info=None, sfreq=None, device="cpu"):
    """
    Factory function to create Braindecode Pretrained Foundation Models.
    """
    if chs_info is None:
        raise ValueError("chs_info must be provided for Interpolated models to compute 3D spatial locations.")
        
    if model_name == "InterpolatedBIOT":
        # Wrap the foundation model with spatial interpolation
        model = bmodels.InterpolatedBIOT.from_pretrained(
            "braindecode/biot-pretrained-six-datasets-18chs", # or "braindecode/biot-pretrained-prest-16chs"
            chs_info=chs_info,
            n_outputs=n_classes,
            n_times=input_window_samples,
            n_chans=in_chans,
            sfreq=sfreq,
            strict=False
        )
    elif model_name == "InterpolatedBENDR":
        model = bmodels.InterpolatedBENDR.from_pretrained(
            "braindecode/braindecode-bendr",
            chs_info=chs_info,
            n_outputs=n_classes,
            n_times=input_window_samples,
            n_chans=in_chans,
            sfreq=sfreq,
            strict=False
        )
    elif model_name == "InterpolatedLaBraM":
        model = bmodels.InterpolatedLaBraM.from_pretrained(
            "braindecode/labram-pretrained",
            chs_info=chs_info,
            n_outputs=n_classes,
            n_times=input_window_samples,
            n_chans=in_chans,
            sfreq=sfreq,
            strict=False
        )
    elif model_name == "InterpolatedSignalJEPA":
        model = bmodels.InterpolatedSignalJEPA.from_pretrained(
            "braindecode/signal-jepa",
            chs_info=chs_info,
            n_outputs=n_classes,
            n_times=input_window_samples,
            n_chans=in_chans,
            sfreq=sfreq,
            strict=False
        )
    elif model_name == "EEGNetv4":
        model = bmodels.EEGNetv4(
            n_chans=in_chans,
            n_outputs=n_classes,
            n_times=input_window_samples
        )
    else:
        raise ValueError(f"Model {model_name} not supported or implemented.")
        
    return model.to(device)
