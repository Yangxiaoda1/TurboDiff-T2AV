import glob
import webdataset as wds
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Optional, Dict


def dict_collation_fn(samples):
    if not samples:
        return {}

    keys = samples[0].keys()
    batched_dict = {key: [] for key in keys}

    for sample in samples:
        for key in keys:
            batched_dict[key].append(sample[key])

    for key in keys:
        if isinstance(batched_dict[key][0], torch.Tensor):
            batched_dict[key] = torch.stack(batched_dict[key])

    return batched_dict


# 获取所有的tar文件，流式解包，读取，重命名，打包成batch。
def create_dataloader(
    tar_path_pattern,  # e.g., "/path/to/dataset/shard_*.tar"
    batch_size,
    num_workers=8,
    shuffle_buffer=1000,
    prefetch_factor=2,
    rename_map: Optional[Dict[str, str]] = None,
):
    shards = glob.glob(tar_path_pattern)  # 获取所有 tar 分片路径
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")

    # Default mapping (T2V): latent.pt/embed.pt/prompt.txt
    # WebDataset wds.rename expects: new_key="old_key"
    if rename_map is None:
        rename_map = dict(
            latents="latent.pt",
            t5_text_embeddings="embed.pt",
            prompts="prompt.txt",
        )

    # 用固定长度包装：每轮恰好 batches_per_epoch 个 batch，不足时重头循环，保证所有 rank 同时 StopIteration
    def make_pipeline():
        return wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.shuffle(1000),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            wds.decode(wds.handle_extension("pt", wds.torch_loads)),
            wds.rename(**rename_map),
            wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
        )

    batches_per_epoch = 8000

    class FixedEpochDataset(IterableDataset):
        """Yields exactly batches_per_epoch batches per rank by cycling the pipeline when needed.
        This keeps all ranks in sync and avoids NCCL collective mismatch (e.g. one rank in
        next epoch forward while others still in backward).
        """

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                nw = worker_info.num_workers
                wid = worker_info.id
                per_worker = batches_per_epoch // nw
                my_limit = per_worker + (batches_per_epoch % nw if wid == 0 else 0)
            else:
                my_limit = batches_per_epoch
            count = 0
            while count < my_limit:
                for batch in make_pipeline():
                    yield batch
                    count += 1
                    if count >= my_limit:
                        return

    dataset = FixedEpochDataset()

    dl_kwargs = dict(
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **dl_kwargs)

    return dataloader
