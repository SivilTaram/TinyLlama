import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
import yaml

# model_name = "tiny_LLaMA_mistral_120M"
model_name = "tiny_LLaMA_vocab_1b"

# Experimental settings
reset_embedding = False
dynamic_weight = False

# Hyperparameters
num_of_devices = 8
global_batch_size = 512
learning_rate = 4e-4
micro_batch_size = 4
max_step = 10000
warmup_steps = 1000
log_step_interval = 10
eval_iters = 100
save_step_interval = 2000
eval_step_interval = 100
# -100 is the default ignore index
# ignore_token_id = -100

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 1e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps


max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Be careful about the weights, it should be something as the len(dataset) * actual reweighting
train_data_config = [
    ("train_cleaned_cc100_ind", 5),
    ("train_redpajama_20b", 1.5)
]

val_data_config = [
    [
        ("valid_cleaned_cc100_ind", 1.0),
    ],
    [
        ("valid_redpajama_20b", 1.0),
    ]
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
# get a random name
random_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
logger = step_csv_logger("out", random_name, flush_logs_every_n_steps=log_iter_interval)
# log hyper-parameters into wandb
wandb_logger = WandbLogger()
wandb_logger.log_hyperparams(hparams)

def setup(
    devices: int = num_of_devices,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    data_yaml_file: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    out_name: str = "default_model",
    load_from: Optional[Path] = None,
) -> None:
    # may modify the global train_data_config
    global train_data_config
    global val_data_config
    
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                sharding_strategy="SHARD_GRAD_OP",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    
    # if train config exists as a yaml file, load it
    if data_yaml_file is not None:
        data_yaml_file = Path(data_yaml_file)
        if data_yaml_file.exists():
            fabric.print("loading config from {}".format(data_yaml_file))
            with open(data_yaml_file, "r") as f:
                # template yaml file is as
                # train_file: weight
                config = yaml.safe_load(f)
            if "train" in config:
                train_config = []
                for k, v in config["train"].items():
                    train_config.append((k, v))
                # update the config
                train_data_config = train_config
            if "valid" in config:
                val_config = []
                for k, v in config["valid"].items():
                    # TODO: by deafult we use separate validation set
                    val_config.append([(k, v)])
                val_data_config = val_config
            # see if any local variable is in the config, if so, update it
            for k, v in config.items():
                if k in globals():
                    fabric.print("update {} to {}".format(k, v))
                    globals()[k] = v
        else:
            fabric.print("config {} does not exist ".format(data_yaml_file))
            # exit the program
            exit(0)
    # log the train & val data config
    wandb_logger.log_hyperparams({"train_data_config": train_data_config, "val_data_config": val_data_config})
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume, out_name, load_from)


def main(fabric, train_data_dir, val_data_dir, resume, out_name, load_from):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)
    out_dir = Path("checkpoints") / out_name

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_dataloader, val_dataloaders = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )

    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    if val_dataloaders is not None:
        for i in range(len(val_dataloaders)):
            val_dataloaders[i] = fabric.setup_dataloaders(val_dataloaders[i])

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True if num_of_devices > 1 else False):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))
        # load pretrained model
        if load_from is not None:
            # use torch.load to load the model
            print("loading model from {}".format(load_from))
            state_dict = torch.load(load_from, map_location=fabric.device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True, assign=True)
        # if reset embedding, reset the embedding layer
        if reset_embedding:
            torch.nn.init.normal_(model.transformer.wte.weight,
                                  mean=0.0, std=math.sqrt(2.0 / 5 / model.transformer.wte.weight.size(1)))
            # reset the output layer, also
            torch.nn.init.normal_(model.lm_head.weight,
                                  mean=0.0, std=math.sqrt(2.0 / 5 / model.lm_head.weight.size(1)))
            for n, p in model.named_parameters():
                if "wte" not in n and "lm_head" not in n:
                    p.requires_grad = False
                else:
                    print("resetting {}".format(n))
                    p.requires_grad = True


    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))
        
    if resume:
        # take the last checkpoint
        resume = resume[-1]
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloaders, monitor, resume, out_dir)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloaders, monitor, resume, out_dir):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloaders is not None:
        # for i in range(len(val_dataloaders)):
        #     validate(fabric, model, val_dataloaders[i])  # sanity check
        # validate(fabric, model, val_dataloaders[0])
        pass

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    
    initial_iter = state["iter_num"]
    curr_iter = 0
    init_english_distribution = {"train_doremi_sample_redpajama_en": 1.0}
    # find the value of the key in the train_data_config
    init_distribution = [0.0] * len(train_data_config)
    for i in range(len(train_data_config)):
        if train_data_config[i][0] in init_english_distribution:
            init_distribution[i] = init_english_distribution[train_data_config[i][0]]
            break
    final_distribution = [0.0] * len(train_data_config)
    for i in range(len(train_data_config)):
        final_distribution[i] = train_data_config[i][1]
    
    loss_func = FusedCrossEntropyLoss()
    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                fabric.print("skip iter {}".format(curr_iter))
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break
        
        if dynamic_weight and state["iter_num"] < warmup_iters:
            # update the weight of the dataset
            weight = [init + (final - init) * state["iter_num"] / warmup_iters for init, final in zip(init_distribution, final_distribution)]
            # normalize the weight into sum to 1
            sum_weight = sum(weight)
            weight = [el / sum_weight for el in weight]
            # update the weight in the train_loader
            train_dataloader.dataset._weights = weight
            fabric.print("update weight to {}".format(weight))
            
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
                f"iter {state['iter_num']} learning rate {lr} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )
 
        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

            
        if val_dataloaders is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            try:            
                val_names = [name[0][0].replace("valid_", "") for name in val_data_config]
            except Exception as e:
                print(e)
                val_names = [str(i) for i in range(len(val_dataloaders))]
            for i in range(len(val_dataloaders)):
                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloaders[i], val_names[i])
                t1 = time.perf_counter() - t0
                monitor.eval_end(t1)
                fabric.print(f"step {state['iter_num']}: {val_names[i]} val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
                fabric.log_dict({f"metric/{val_names[i]}_val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                fabric.log_dict({f"metric/{val_names[i]}_val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                fabric.barrier()

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['step_count']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, name: str = None) -> torch.Tensor:
    fabric.print(f"Validating {name} ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        # fabric.print("val data: {}".format(val_data))
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
    
        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()
    
    # print top 100 losses
    out = losses.mean()
    model.train()
    return out


def create_train_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_val_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    
    val_data_loaders = []
    for data_config in val_data_config:
        datasets = []
        for prefix, _ in data_config:
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
            random.seed(seed)
            random.shuffle(filenames)

            dataset = PackedDataset(
                filenames,
                # n_chunks control the buffer size. 
                # Note that the buffer size also impacts the random shuffle
                # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
                n_chunks=1,
                block_size=block_size,
                shuffle=shuffle,
                seed=seed+fabric.global_rank,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
            )
            datasets.append(dataset)

        if not datasets:
            raise RuntimeError(
                f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
            )

        weights = [weight for _, weight in data_config]
        sum_weights = sum(weights)
        weights = [el / sum_weights for el in weights]

        check_flag = True
        for dataset in datasets:
            if len(dataset._filenames) == 0:
                check_flag = False
                break
        
        if check_flag:
            combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
            val_data_loaders.append(DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True))
        else:
            fabric.print("skip val dataset {}".format(data_config))
    return val_data_loaders


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_train_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_val_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
