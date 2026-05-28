import braindecode.models as bmodels

def create_model(model_name, n_classes, in_chans, input_window_samples, device="cpu"):
    """
    Factory function to create Braindecode models.
    """
    if model_name == "EEGConformer":
        model = bmodels.EEGConformer(
            n_outputs=n_classes,
            n_chans=in_chans,
            n_times=input_window_samples
        )
    elif model_name == "EEGNetv4":
        model = bmodels.EEGNetv4(
            n_outputs=n_classes,
            n_chans=in_chans,
            n_times=input_window_samples,
            final_conv_length="auto"
        )
    elif model_name == "ShallowFBCSPNet":
        model = bmodels.ShallowFBCSPNet(
            n_outputs=n_classes,
            n_chans=in_chans,
            n_times=input_window_samples,
            final_conv_length="auto"
        )
    elif model_name == "Deep4Net":
        model = bmodels.Deep4Net(
            n_outputs=n_classes,
            n_chans=in_chans,
            n_times=input_window_samples,
            final_conv_length="auto"
        )
    else:
        raise ValueError(f"Model {model_name} not supported or implemented.")
        
    return model.to(device)
