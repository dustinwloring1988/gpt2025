import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import numpy as np

# Import the optimized model
from enhanced_gpt2_model import OptimizedGPT, OptimizedGPTConfig, CosineAnnealingWithWarmup

# Import evaluation utilities (assuming hellaswag.py is available)
try:
    from hellaswag import render_example, iterate_examples
    HELLASWAG_AVAILABLE = True
except ImportError:
    HELLASWAG_AVAILABLE = False
    print("HellaSwag evaluation not available")

# -----------------------------------------------------------------------------
# Enhanced Data Loader with optimizations
# -----------------------------------------------------------------------------

def load_tokens(filename):
    """Load tokens from .npy file"""
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class OptimizedDataLoaderLite:
    """Enhanced data loader with memory optimization and prefetching"""
    
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Get shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        
        # Optimization: preload multiple shards to reduce I/O
        self.prefetch_shards = min(2, len(shards))  # Prefetch up to 2 shards
        self.shard_cache = {}
        
        self.reset()

    def reset(self):
        """Reset to beginning of dataset"""
        self.current_shard = 0
        self.current_position = self.B * self.T * self.process_rank
        
        # Load initial shard(s)
        self._load_shard(self.current_shard)
        
        # Prefetch next shard if available
        if self.prefetch_shards > 1 and len(self.shards) > 1:
            next_shard = (self.current_shard + 1) % len(self.shards)
            self._prefetch_shard(next_shard)

    def _load_shard(self, shard_idx):
        """Load a specific shard"""
        if shard_idx not in self.shard_cache:
            self.shard_cache[shard_idx] = load_tokens(self.shards[shard_idx])
        self.tokens = self.shard_cache[shard_idx]

    def _prefetch_shard(self, shard_idx):
        """Prefetch a shard in background"""
        if shard_idx not in self.shard_cache and len(self.shard_cache) < self.prefetch_shards:
            self.shard_cache[shard_idx] = load_tokens(self.shards[shard_idx])

    def next_batch(self):
        """Get next batch of data"""
        B, T = self.B, self.T
        
        # Get current batch
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        
        # Advance position
        self.current_position += B * T * self.num_processes
        
        # Check if we need to move to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # Clean up old shard to save memory
            if len(self.shard_cache) >= self.prefetch_shards:
                old_shard = (self.current_shard - self.prefetch_shards) % len(self.shards)
                if old_shard in self.shard_cache and old_shard != self.current_shard:
                    del self.shard_cache[old_shard]
            
            # Move to next shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self._load_shard(self.current_shard)
            self.current_position = B * T * self.process_rank
            
            # Prefetch next shard
            if self.prefetch_shards > 1:
                next_shard = (self.current_shard + 1) % len(self.shards)
                self._prefetch_shard(next_shard)
        
        return x, y

# -----------------------------------------------------------------------------
# Enhanced HellaSwag evaluation with optimizations
# -----------------------------------------------------------------------------

def get_most_likely_row(tokens, mask, logits):
    """Optimized HellaSwag evaluation function"""
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# Enhanced training loop with all optimizations
# -----------------------------------------------------------------------------

def main():
    # Setup distributed training
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if master_process:
            print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Set random seeds
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Enhanced model configuration
    config = OptimizedGPTConfig(
        # Model architecture
        block_size=1024,
        vocab_size=50304,  # Padded for efficiency
        n_layer=12,
        n_head=12,
        n_embd=768,
        
        # Position encoding strategy
        position_encoding_type='hybrid',  # Try 'rope', 'nope', 'hybrid', 'learned'
        hybrid_ratio=0.5,  # 50% RoPE, 50% NoPE
        rope_base=10000,
        
        # Optimizations
        use_rmsnorm=True,
        activation_function='swiglu',  # More efficient than GELU
        attention_bias=False,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        layer_norm_eps=1e-6,
        
        # Memory optimizations
        use_flash_attention=True,
        use_gradient_checkpointing=False,  # Enable for larger models
    )

    # Training hyperparameters
    total_batch_size = 524288  # 2**19, ~0.5M tokens
    B = 16  # Reduced micro batch size for memory efficiency with optimizations
    T = 1024  # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Enhanced data loaders
    train_loader = OptimizedDataLoaderLite(
        B=B, T=T, 
        process_rank=ddp_rank, 
        num_processes=ddp_world_size, 
        split="train"
    )
    val_loader = OptimizedDataLoaderLite(
        B=B, T=T, 
        process_rank=ddp_rank, 
        num_processes=ddp_world_size, 
        split="val"
    )

    # Set precision for better performance
    torch.set_float32_matmul_precision('high')

    # Create model
    if master_process:
        print("Creating optimized model...")
        print(f"Position encoding: {config.position_encoding_type}")
        print(f"Hybrid ratio: {config.hybrid_ratio}")
        print(f"Using RMSNorm: {config.use_rmsnorm}")
        print(f"Activation: {config.activation_function}")

    model = OptimizedGPT(config)
    model.to(device)
    
    # Optional: Load from pretrained GPT-2
    # model = OptimizedGPT.from_pretrained("gpt2", override_args={
    #     'position_encoding_type': 'hybrid',
    #     'hybrid_ratio': 0.5,
    #     'use_rmsnorm': True,
    #     'activation_function': 'swiglu'
    # })
    # model.to(device)

    # Compile model for better performance (optional)
    use_compile = False  # Set to True if you want to use torch.compile
    if use_compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Setup DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Enhanced learning rate schedule
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073
    
    lr_scheduler = CosineAnnealingWithWarmup(
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps
    )

    # Configure optimizer with enhanced settings
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, 
        learning_rate=max_lr, 
        device_type=device_type,
        beta1=0.9,
        beta2=0.95
    )

    # Logging setup
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass

    # Training loop with enhanced monitoring
    if master_process:
        print(f"Starting training for {max_steps} steps...")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Validation evaluation
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    
                    # Use mixed precision for memory efficiency
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                
                # Save checkpoints
                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        # HellaSwag evaluation
        if (step % 250 == 0 or last_step) and HELLASWAG_AVAILABLE and (not use_compile):
            model.eval()
            num_correct_norm = 0
            num_total = 0
            
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            
            # Reduce stats across processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # Text generation sampling
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            raw_model.reset_cache()  # Reset KV cache
            
            enc = tiktoken.get_encoding("gpt2")
            num_return_sequences = 4
            max_length = 32
            
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            
            with torch.no_grad():
                while xgen.size(1) < max_length:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen[:, -1:], use_cache=True)  # Only process last token
                    
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    
                    # Top-k sampling
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            
            # Print generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                if master_process:
                    print(f"rank {ddp_rank} sample {i}: {decoded}")

        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # Sync gradients only on last micro step
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update learning rate
        lr = lr_scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        # Wait for GPU to finish
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            # Calculate MFU
            mfu = raw_model.estimate_mfu(grad_accum_steps * ddp_world_size, dt)
            
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | MFU: {mfu*100:.2f}%")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
