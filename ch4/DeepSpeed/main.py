import torch
import argparse
import deepspeed
import math
from utils.ds_utils import get_train_ds_config
from utils.utils import set_random_seed, to_device, get_all_reduce_mean, get_optimizer_grouped_parameters, print_rank_0, save_hf_format, save_zero_three_model
from utils.model.model_utils import create_llama_model
from utils.data.data_utils import create_prompt_dataset
from utils.perf import print_throughput
from utils.module.lora import convert_lora_to_linear_layer
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.data.distributed import DistributedSampler

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from transformers import AutoModelForCausalLM, SchedulerType, default_data_collator, get_scheduler

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_args()

    ### Initialize DeepSpeed ###
    # Using local_rank for setting device
    if args.local_rank == -1:
        # not using distributed training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # using distributed training
        torch.cuda.set_device(args.local_rank) # bind current process to a specific GPU
        device = torch.device("cuda", args.local_rank) # create device object for the specific GPU

    # Initialize backend
    # torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()  # Initialize DeepSpeed distributed environment

    # Get the global rank of the current process
    args.global_rank = torch.distributed.get_rank()

    # Get the configuration for DeepSpeed
    ds_config = get_train_ds_config(offload=args.offload, # whether to offload parameters to CPU
                                    stage=args.zero_stage, # ZeRO stage
                                    enable_tensorboard=args.enable_tensorboard, # whether to enable TensorBoard
                                    tb_path=args.tensorboard_path, # TensorBoard log path
                                    tb_name="step1_model" # TensorBoard log name
                                    )
    
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    # Calculate the total train batch size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # set training seed
    set_random_seed(args.seed)

    torch.distributed.barrier()  # Ensure all processes are synchronized before starting training

    ### Load model and tokenizer ###
    model, tokenizer = create_llama_model(args.model_name_or_path)

    ### training data ###
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len
    )

    # sampler
    if args.localrank == -1: # not using distributed training
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = RandomSampler(eval_dataset)
    else: # using distributed training
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size
    )

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch) # unpack the batch dictionary
            
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)

        try:
            losses = get_all_reduce_mean(losses) # average the loss across all processes
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float('inf')
        
        return perplexity, losses.item()
    
    ### Optimizer Settings ###
    # divide the model parameters into two groups: weight decay and no weight decay
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,
        args.weight_decay,
        args.learning_rate
    )

    # if offload to CPU, using DeepSpeedCPUAdam, otherwise using FusedAdam
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam

    optimizer = AdamOptimizer(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    ) # round up to an integer

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, # type of learning rate scheduler
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps, # number of warmup steps
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch, # total number of training steps
    )

    # Initialization by DeepSpeed
    model, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True # initialize distributed training
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable() # enable checkpointing

    ### Training ###
    print_rank_0("*** Running Training ***", args.global_rank)
    print_rank_0(f"*** Evaluating Perplexity, Epoch {0}/{args.num_train_epochs} ***", args.global_rank)

    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, \
            Total Micro Batches {len(train_dataloader)}",
            args.global_rank
        )
        model.train()

        import time
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device) # move the batch to the appropriate device
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            if args.print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, \
                        Rank: {torch.distributed.get_rank()}, loss: {loss}"
                )
            
            model.backward(loss)
            model.step()
            end = time.time()

            # calculate throughput
            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start, args.global_rank)

        # save model
        if args.output_dir is not None:
            print_rank_0("Saving the final model ...", args.global_rank)
            model = convert_lora_to_linear_layer(model)

            if args.global_rank == 0:
                save_hf_format(model, tokenizer, args) # save model in Hugging Face format
            
            if args.zero_stage == 3:
                save_zero_three_model(model, args.global_rank, args.output_dir, zero_stage=args.zero_stage) # save model in ZeRO-3 format


if __name__ == "__main__":
    main()