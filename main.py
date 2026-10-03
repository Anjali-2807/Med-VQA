import os
import numpy as np
if not hasattr(np, "Inf"): np.Inf = np.inf
if not hasattr(np, "NaN"): np.NaN = np.nan
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "bool"): np.bool = bool

import copy
import os
import resource

import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.modules import M3AETransformerSS

# from m3ae.datasets.ROCO import ROCOFeatureDataset
# from torch.utils.data import DataLoader

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


@ex.automain
def main(_config):
    _config = {
        k: v for k, v in _config.items()
        if isinstance(v, (int, float, str, bool, list, dict, tuple, set, type(None)))
    }
    pl.seed_everything(_config["seed"])

    # Data modules
    # Only use distributed sampler when actually running multi-GPU DDP
    _use_dist = _config["num_nodes"] * (
        _config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"])
    ) > 1
    dm = MTDataModule(_config, dist=_use_dist)

    # retrieval
    # 创建检索数据集
    # retrieval_function = None
    # if "retrieval" in _config and _config["retrieval"]:
    #     # if "retrieval_dataset" in _config:
    #     retrieval_dataset = ROCOFeatureDataset("train", _config['retrieval_data'], device=f"cuda:{_config['gpu_ids']}")
    #     # retrieval_dataset = ROCOFeatureDataset("train", _config['retrieval_data'], device="cpu")
    #     # retrieval_dataset = load_dataset(_config["datafolder"], _config["retrieval_dataset"], "train", device)
    #     retrieval_loader = DataLoader(retrieval_dataset, _config["retrieval_batch"], shuffle=True, num_workers=2)
    #
    #     if "k" in _config:
    #         k = _config["k"]
    #     else:
    #         k = 15
    #
    #     print(f"Using {k}-nn retrieval from {retrieval_dataset.dataroot} with only training data ...")
    #     retrieval_dataset.create_retrieval_dataset(retrieval_loader, is_training_phase=True, retrieval_k=k)
    #     retrieval_function = retrieval_dataset.retrieve_closest_qa_pairs

    # Module
    model = M3AETransformerSS(_config)  # 网络模块初始化
    # model.set_retrieval_function(retrieval_function)
    # print(model)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    # wb_logger = pl.loggers.WandbLogger(project="MICCAI-M3AE", name=run_name)
    loggers = [tb_logger]

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False
    )  # 创建回调函数，保存最佳模型
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    # print(_config)
    gpu_ids = (_config['gpu_ids'])
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    # PL 1.9 requires -1 (not None) to mean "no max_steps limit"
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else -1
    max_epochs = _config["max_epoch"] if max_steps == -1 else 1000

    # Trainer
    import torch
    use_gpu = torch.cuda.is_available() and num_gpus > 0
    accelerator = "gpu" if use_gpu else "cpu"
    devices = [gpu_ids] if (use_gpu and isinstance(gpu_ids, int)) else (gpu_ids if use_gpu else "auto")
    strategy = "ddp" if (use_gpu and num_gpus > 1) else None

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "num_nodes": _config["num_nodes"],
        "precision": _config["precision"],
        "deterministic": True,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "callbacks": callbacks,
        "logger": loggers,
        "accumulate_grad_batches": grad_steps,
        "log_every_n_steps": 10,
        "fast_dev_run": _config["fast_dev_run"],
        "val_check_interval": _config["val_check_interval"],
        "default_root_dir": _config["default_root_dir"],
    }
    if strategy:
        trainer_kwargs["strategy"] = strategy

    trainer = pl.Trainer(**trainer_kwargs)

    resume_from = _config.get("resume_from", None)
    if resume_from == "":
        resume_from = None

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=resume_from)
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
    else:
        trainer.test(model, datamodule=dm, ckpt_path=resume_from)
        # get_cam(model)

