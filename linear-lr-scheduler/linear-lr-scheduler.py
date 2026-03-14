def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:

    # no schedule possible
    if total_steps == 0:
        return final_lr

    # warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        return initial_lr * step / warmup_steps

    # decay phase
    if step <= total_steps:
        decay_steps = total_steps - warmup_steps
        progress = (total_steps - step) / max(1, decay_steps)
        return final_lr + (initial_lr - final_lr) * progress

    # after training
    return final_lr