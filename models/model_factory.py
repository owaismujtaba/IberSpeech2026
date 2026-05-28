import braindecode.models as bmodels

def create_model(model_name, n_classes, in_chans, input_window_samples, chs_info=None, device="cpu"):
    """
    Factory function to create Braindecode Pretrained Foundation Models.
    """
    if chs_info is None:
        raise ValueError("chs_info must be provided for Interpolated models to compute 3D spatial locations.")
        
    if model_name == "InterpolatedBIOT":
        # Wrap the foundation model with spatial interpolation
        model = bmodels.InterpolatedBIOT(
            "braindecode/biot-pretrained-prest-16chs",
            chs_info=chs_info,
            n_outputs=n_classes
        )
    elif model_name == "InterpolatedBENDR":
        model = bmodels.InterpolatedBENDR(
            "braindecode/braindecode-bendr",
            chs_info=chs_info,
            n_outputs=n_classes
        )
    elif model_name == "InterpolatedLaBraM":
        model = bmodels.InterpolatedLaBraM(
            "braindecode/labram-pretrained",
            chs_info=chs_info,
            n_outputs=n_classes
        )
    elif model_name == "InterpolatedSignalJEPA":
        model = bmodels.InterpolatedSignalJEPA(
            "braindecode/SignalJEPA-pretrained",
            chs_info=chs_info,
            n_outputs=n_classes
        )
    else:
        raise ValueError(f"Pretrained Model {model_name} not supported or implemented.")
        
    return model.to(device)
