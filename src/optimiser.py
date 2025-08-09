import math


def lr_cosine_with_warmup(t, t_warmup, T, eta_max, eta_min=1e-6):
    """
    t: current local step for *this shard/worker*
    t_warmup: warmup steps
    T: target total steps for this shard (post-warmup, schedule clamps if t > T)
    eta_max: peak LR reached at end of warmup
    eta_min: floor LR (kept >0 so training still progresses past T)
    """
    if t < t_warmup:
        return eta_max * (t / max(1, t_warmup))
    # Clamp progress in [0, 1] and guard denominator
    denom = max(1, T - t_warmup)
    x = (t - t_warmup) / denom
    x = 0.0 if x < 0 else 1.0 if x > 1 else x
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * x))


class ShardLRScheduler:
    def __init__(self, t_warmup, T, eta_max, eta_min=1e-6):
        self.t = 0
        self.t_warmup = t_warmup
        self.T = T
        self.eta_max = eta_max
        self.eta_min = eta_min
    def step_and_get(self):
        lr = lr_cosine_with_warmup(self.t, self.t_warmup, self.T, self.eta_max, self.eta_min)
        self.t += 1
        return lr

