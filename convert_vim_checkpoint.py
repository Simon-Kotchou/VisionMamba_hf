import argparse
import torch
from pathlib import Path

from transformers import PretrainedConfig
from modeling_vision_mamba import VisionMambaConfig, VisionMambaForImageClassification

def rename_key(name):
    if "patch_embed.proj" in name:
        return name.replace("patch_embed.proj", "vision_mamba.patch_embed.proj")
    elif "pos_embed" in name:
        return name.replace("pos_embed", "vision_mamba.pos_embed")
    elif "cls_token" in name:
        return name.replace("cls_token", "vision_mamba.cls_token")
    elif "layers" in name:
        return name.replace("layers", "vision_mamba.blocks")
    elif "norm_f" in name:
        return name.replace("norm_f", "vision_mamba.norm")
    elif "head" in name:
        return name.replace("head", "classifier")
    else:
        return name

# def combine_bidirectional_params(state_dict):
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         if "mixer" in key:
#             # Remove the '_b' suffix for backward direction parameters
#             new_key = key.replace("mixer.A_b_log", "mixer.A_log")
#             new_key = new_key.replace("mixer.D_b", "mixer.D")
#             new_key = new_key.replace("mixer.conv1d_b", "mixer.conv1d")
#             new_key = new_key.replace("mixer.x_proj_b", "mixer.x_proj")
#             new_key = new_key.replace("mixer.dt_proj_b", "mixer.dt_proj")
            
#             if new_key in new_state_dict:
#                 # Combine forward and backward parameters
#                 new_state_dict[new_key] = torch.cat([new_state_dict[new_key], value], dim=0)
#             else:
#                 new_state_dict[new_key] = value
#         else:
#             new_state_dict[key] = value
#     return new_state_dict

def convert_vim_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, push_to_hub=False):
    # Load the original state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # # Combine bidirectional parameters
    # state_dict = combine_bidirectional_params(state_dict)

    # Load or create config
    if config_path:
        config = PretrainedConfig.from_json_file(config_path)
    else:
        config = VisionMambaConfig(
            img_size=224,
            patch_size=16,
            stride=8,
            num_classes=1000,
            embed_dim=384,  # for Vim-S
            depth=24,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            if_cls_token=True,
            use_middle_cls_token=True,
            bimamba_type="v2",
        )

    # Create the new model
    model = VisionMambaForImageClassification(config)

    # Rename the keys in the state dict
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = rename_key(key)
        new_state_dict[new_key] = value

    # Load the new state dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    # Save the model
    if pytorch_dump_folder_path:
        Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Model saved to {pytorch_dump_folder_path}")

    if push_to_hub:
        model.push_to_hub("Simon-Kotchou/vim-small-imagenet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the original Vim checkpoint")
    parser.add_argument("--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model.")
    parser.add_argument("--config_path", type=str, help="Path to the config.json file. If not provided, default config will be used.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the converted model to the Hub.")
    args = parser.parse_args()

    convert_vim_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.push_to_hub)