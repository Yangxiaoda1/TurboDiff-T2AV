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
                iteration_offset=0,
            )
        ),
    )


"""
Example:

export MOVA_ROOT="/home/jovyan/codes/turbodiff/new_Turbo/MOVA"
torchrun --nproc_per_node=8 -m scripts.train   --config=turbodiffusion/rcm/configs/registry_rcm.py   -- experiment=mova_360p_t2a_rcm   model=ddp_t2a_distill_rcm   dataloader_train.tar_path_pattern="/data/datasets/turbodiff_datasets_and_ckpt/t2a_latents/shard_*.tar"   model.config.mova_ckpt_path="/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p"
"""

MOVA_360P_T2A_RCM: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "distill"},
            {"override /data_train": "webdataset"},
            {"override /model": "ddp_t2a_distill_rcm"},
            {"override /callbacks": ["basic", "dataloading_speed", "wandb"]},
            {"override /checkpoint": "local"},
            {"override /optimizer": "fusedadamw"},
            {"override /ckpt_type": "dcp_distill"},
            "_self_",
        ],
        job=dict(
            group="RCM_MOVA",
            name="mova_360p_t2a_rcm",
        ),
        optimizer=dict(
            lr=2e-6,
            weight_decay=0.01,
            betas=(0.0, 0.999),
        ),
        scheduler=dict(
            warm_up_steps=[10],
            cycle_lengths=[10_000_000_000_000],
            f_start=[1.0e-2],
            f_max=[1.0],
            f_min=[1.0],
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            run_validation=False,
            ddp=dict(
                find_unused_parameters=True,
                static_graph=False,
            ),
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
                teacher_ckpt_path="/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p",
                student_ckpt_path="/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p",
                precision="bfloat16",
                rectified_flow_t_scaling_factor=1000.0,
                student_num_layers_audio=0,
                iteration_offset=0,
                student_update_freq=5,
                max_simulation_steps_fake=4,
                p_mean=-0.8,
                p_std=1.6,
                p_D_mean=0.0,
                p_D_std=1.6,
                timestep_shift=5.0,
                fd_type=0,
                fd_size=1e-4,
                tangent_warmup=1000,
                loss_scale=100.0,
                loss_scale_dmd=1.0,
                loss_scale_fake_score=1.0,
                loss_scale_teacher=0.0,
                fake_score_lr=4e-7,
                fake_score_weight_decay=0.01,
                fake_score_betas=(0.0, 0.999),
                teacher_guidance=5.0,
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
cs.store(group="experiment", package="_global_", name="mova_360p_t2a_rcm", node=MOVA_360P_T2A_RCM)
cs.store(group="experiment", package="_global_", name="mova_360p_t2a_rcm_debug", node=build_debug_run(MOVA_360P_T2A_RCM))
