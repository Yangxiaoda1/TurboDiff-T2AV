# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyDict


"""
Example:

export MOVA_ROOT="/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/MOVA"
torchrun --nproc_per_node=8 -m scripts.train \
  --config=turbodiffusion/rcm/configs/registry_sla.py \
  -- experiment=mova_360p_it2av_sla \
  dataloader_train.tar_path_pattern="/path/to/it2av_tar/shard_*.tar" \
  model.config.teacher_ckpt_path="/apdcephfs_gy2/share_302507476/xiaodayang/MOVA/MOVA-360p" \
  model.config.student_num_layers_video=16 \
  model.config.student_num_layers_audio=16
"""

MOVA_360P_IT2AV_SLA: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "standard"},
            {"override /data_train": "webdataset"},
            {"override /model": "ddp_it2av_distill_sla"},
            {"override /callbacks": ["basic", "dataloading_speed", "wandb"]},
            {"override /checkpoint": "local"},
            "_self_",
        ],
        job=dict(
            group="SLA_MOVA",
            name="mova_360p_it2av_sla",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            run_validation=False,
        ),
        # WebDataset: map tar filenames -> batch keys expected by IT2AVDistillModel_SLA
        dataloader_train=dict(
            tar_path_pattern="/path/to/it2av_tar/shard_*.tar",
            batch_size=1,
            num_workers=8,
            shuffle_buffer=1000,
            prefetch_factor=2,
            rename_map=dict(
                v_latents="v_latent.pt",
                a_latents="a_latent.pt",
                ref_latents="ref.pt",
                t5_text_embeddings="embed.pt",
                prompts="prompt.txt",
            ),
        ),
        model=dict(
            config=dict(
                teacher_ckpt_path="/apdcephfs_gy2/share_302507476/xiaodayang/MOVA/MOVA-360p",
                precision="bfloat16",
                rectified_flow_t_scaling_factor=1000.0,
                # Keep student architecture unchanged by default (0 = no truncation).
                # If you need to reduce memory, override via CLI.
                student_num_layers_video=0,
                student_num_layers_audio=0,
                # Enable teacher CFG during distillation (uncond + scale * (cond - uncond)).
                teacher_guidance=5.0,
                use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=False,
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name="mova_360p_it2av_sla", node=MOVA_360P_IT2AV_SLA)
