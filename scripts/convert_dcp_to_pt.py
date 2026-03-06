import torch
import torch.distributed.checkpoint as dcp
import sys
import os

# Add paths
sys.path.append(os.getcwd())
sys.path.append("turbodiffusion")
sys.path.append("/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/MOVA")

from rcm.models.it2av_model_distill_rcm import IT2AVDistillModel_rCM, IT2AVDistillConfig_rCM

def convert(dcp_path, output_path):
    print(f"Loading DCP from {dcp_path}...")
    
    # 1. Create model shell
    # Use CPU to save GPU memory
    config = IT2AVDistillConfig_rCM(
        teacher_ckpt_path="/apdcephfs_gy2/share_302507476/xiaodayang/MOVA/MOVA-360p",
        student_ckpt_path="",
        fsdp_shard_size=1, # Disable FSDP wrapping for conversion
        # Set dummy paths to avoid actual loading if possible, but the code calls from_pretrained immediately
    )
    
    print("Initializing model structure (this may take time and RAM)...")
    # Hack: set device to cpu globally if possible, model init uses config.precision but to() logic
    # The model __init__ does `to(device)` for students. 
    # We rely on torch being smart enough or we just run this on a node with GPU.
    model = IT2AVDistillModel_rCM(config)
    
    # 2. Load DCP
    print("Loading distributed checkpoint...")
    
    # Force point to 'model' subdirectory if it exists
    # The metadata is located inside 'iter_xxx/model/.metadata'
    if not dcp_path.endswith("model") and os.path.isdir(os.path.join(dcp_path, "model")):
        dcp_path = os.path.join(dcp_path, "model")
        print(f"Adjusted checkpoint path to model subdirectory: {dcp_path}")
    
    # Since we are loading from the 'model' subdir, the metadata describes the model state dict directly.
    # We should NOT wrap it in {"model": ...}
    state_dict = model.state_dict()
    
    try:
        dcp.load(state_dict=state_dict, checkpoint_id=dcp_path)
        print("Loaded successfully.")
    except Exception as e:
        print(f"Failed to load DCP: {e}")
        print("Tip: Ensure the path points to a directory containing .metadata file.")
        raise e

    # 3. Extract and Renaming
    # The pipeline expects keys matching 'video_dit' and 'audio_dit' (if we load into pipe directly)
    # OR we can just save 'student_video' and 'student_audio' and let inference script handle it.
    # Let's save as 'video_dit' and 'audio_dit' to be ready for MOVA pipeline.
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("student."):
            # Remove "student." prefix
            # student.video_dit.xxx -> video_dit.xxx
            # student.audio_dit.xxx -> audio_dit.xxx
            # student.bridge.xxx    -> bridge.xxx
            new_k = k.replace("student.", "")
            clean_state_dict[new_k] = v.cpu()
            
    print(f"Extracted {len(clean_state_dict)} keys from Student model.")
    
    # 4. Save
    torch.save(clean_state_dict, output_path)
    print(f"Saved converted checkpoint to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert_dcp_to_pt.py <dcp_folder> <output.pt>")
        sys.exit(1)
        
    convert(sys.argv[1], sys.argv[2])
