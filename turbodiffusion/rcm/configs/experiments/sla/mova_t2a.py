from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyDict


def build_debug_run(job):
    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=dict(
            max_iter=25,
            logging_iter=2,
        ),
        checkpoint=dict(
            save_iter=10,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                tangent_warmup=8,
            )
        ),
    )


"""
Example:

export MOVA_ROOT="/home/jovyan/codes/turbodiff/new_Turbo/MOVA"
torchrun --nproc_per_node=8 -m scripts.train \
  --config=turbodiffusion/rcm/configs/registry_sla.py \
  -- experiment=mova_360p_t2a_sla \
  dataloader_train.tar_path_pattern="/data/datasets/turbodiff_datasets_and_ckpt/t2a_latents/shard_*.tar" \
  model.config.mova_ckpt_path="/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p"
"""

MOVA_360P_T2A_SLA: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "standard"},
            {"override /data_train": "webdataset"},
            {"override /model": "ddp_t2a_distill_sla"},
            {"override /callbacks": ["basic", "dataloading_speed", "wandb"]},
            {"override /checkpoint": "local"},
            "_self_",
        ],
        job=dict(
            group="SLA_MOVA",
            name="mova_360p_t2a_sla",
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
        dataloader_train=dict(
            tar_path_pattern="/data/datasets/turbodiff_datasets_and_ckpt/t2a_latents/shard_*.tar",
            batch_size=2,
            num_workers=8,
            shuffle_buffer=1000,
            prefetch_factor=2,
            rename_map=dict(
                a_latents="a_latent.pt",
                t5_text_embeddings="embed.pt",
                prompts="prompt.txt",
            ),
        ),
        model=dict(
            config=dict(
                mova_ckpt_path="/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p",
                precision="bfloat16",
                rectified_flow_t_scaling_factor=1000.0,
                student_num_layers_audio=0,
                fd_size=1e-3,
                tangent_warmup=1000,
                loss_scale=100.0,
                teacher_guidance=0.0,
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
cs.store(group="experiment", package="_global_", name="mova_360p_t2a_sla", node=MOVA_360P_T2A_SLA)
cs.store(group="experiment", package="_global_", name="mova_360p_t2a_sla_debug", node=build_debug_run(MOVA_360P_T2A_SLA))

