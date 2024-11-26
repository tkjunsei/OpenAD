import torch


checkpoint = torch.load("./log/openad_pn2/OPENAD_PN2_FULL_SHAPE_Release_11231539/best_model.t7")
for k, v in checkpoint.items():
    print(f"Layer: {k}, Shape: {v.shape if hasattr(v, 'shape') else 'No shape'}")
