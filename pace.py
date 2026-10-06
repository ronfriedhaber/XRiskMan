"""Batched PACE practice dynamics. Money is in billions; time is in seconds.

Independent NumPy implementation of the public client, not the multiplayer server.
No network, wall clock, rendering, or automatic resets. See README for equations.
"""
from config import EnvironmentConfig as Config

import numpy as np


def speed_limit(x):
    early = -np.expm1(-np.maximum(x, 0) / 12)
    late = -np.expm1(-(np.maximum(x - 24, 0) / 12) ** 2)
    return 1.5 + (3.24 - 1.5) * (0.08 * early + 0.92 * late)


def cautious(env):
    stopping = env.v[:, 0]**2 / (2 * env.c.deceleration * speed_limit(env.x[:, 0]))
    return (env.x[:, 0] + stopping < env.safety - np.where(env.held[:, 0], 0, .3)).astype(np.int64)


def profit(deployed):
    """Annual profit for both labs; logaddexp avoids unstable exp/softplus."""
    gap = deployed - deployed[:, ::-1]
    base = 5 + 10 * np.log1p(0.025 * deployed)
    lag = np.maximum(-gap, 0)
    discounted = base / (1 + 0.4 * np.log1p(lag / 2))
    pressure = np.logaddexp(0, np.log(np.maximum(lag, 1e-300)) + (deployed[:, ::-1] - 22) / 4)
    blend = np.minimum(1, lag / 2)
    blend = blend**2 * (3 - 2 * blend)
    trailing = discounted + ((discounted + 8) / (1 + pressure) - 8 - discounted) * blend
    leading = base + 3 * np.log1p(np.maximum(gap, 0) / 2)
    return np.where(gap >= 0, leading, trailing)


def hazard(deployed, safety, maximum=0.24):
    capability = deployed.max(axis=1)
    gap = np.maximum(capability - safety, 0)
    # Equivalent to the client's reciprocal formula, including at zero.
    return maximum * capability**2 / (capability**2 + 144) * gap**2 / (gap**2 + 75)


def _uniform(state, mask):
    """Exact uint32 Mulberry32 stream used by the client; advance selected rows."""
    bits = np.uint64(0xFFFFFFFF)
    s = (state[mask] + np.uint64(1831565813)) & bits
    state[mask] = s
    t = ((s ^ (s >> 15)) * (s | 1)) & bits
    t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & bits)) & bits
    return ((t ^ (t >> 14)) & bits).astype(np.float64) / 2**32


class Pace:
    """N games, two labs. Actions: 0=release/private, 1=hold/private,
    2=release/share, 3=hold/share. Inputs have shape (N, 2).

    With bot=True, lab 1 uses the original per-physics-tick bot and delayed
    sharing reciprocity. Done rows freeze until explicitly reset.
    """
    obs_dim = 16

    def __init__(self, seeds, config=Config(), *, legacy=False):
        self.c = config
        self.legacy = legacy
        seeds = np.asarray(seeds)
        if seeds.ndim != 1 or not len(seeds):
            raise ValueError("seeds must be a nonempty vector")
        self.n = len(seeds)
        self.lag = round(config.delay / config.dt)
        self.horizon = round(config.duration / config.dt)
        self.repeat = round(config.decision_dt / config.dt)
        self.cursor = 0
        self.queue = np.zeros((self.lag + 1, self.n, 2))
        for name in ("x", "v", "d", "cash", "p"):
            setattr(self, name, np.zeros((self.n, 2)))
        self.held = np.zeros((self.n, 2), dtype=bool)
        self.sharing = self.held.copy()
        self.last_move = np.full((self.n, 2), -self.lag, dtype=np.int64)
        for name in ("ticks", "plateau_phase"):
            setattr(self, name, np.zeros(self.n, dtype=np.int64))
        for name in ("safety", "safety_speed", "target", "next_change", "H", "threshold",
                     "plateau_elapsed", "plateau_duration", "reply_at"):
            setattr(self, name, np.zeros(self.n))
        self.reply_value = np.zeros(self.n, dtype=bool)
        self.done = np.zeros(self.n, dtype=bool)
        self.crashed = self.done.copy()
        self.random = np.zeros(self.n, dtype=np.uint64)
        self.plateau_random = self.random.copy()
        self.reset(seeds)

    def reset(self, seeds, mask=None):
        mask = np.ones(self.n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        seeds = np.asarray(seeds)
        if mask.shape != (self.n,) or seeds.shape != (int(mask.sum()),):
            raise ValueError("provide exactly one seed per reset row")
        if seeds.dtype.kind not in "iu" or np.any(seeds < 0) or np.any(seeds > 0xFFFFFFFF):
            raise ValueError("seeds must be uint32 integers")
        seed = seeds.astype(np.uint64)
        u = seed.copy()
        u ^= (u << 13) & np.uint64(0xFFFFFFFF)
        u ^= u >> 17
        u ^= (u << 5) & np.uint64(0xFFFFFFFF)
        self.threshold[mask] = -np.log((u.astype(np.float64) + 1) / (2**32 + 1))
        self.random[mask] = seed ^ np.uint64(2654435769)
        self.plateau_random[mask] = seed ^ np.uint64(2246822507)
        for name in ("x", "v", "d", "cash", "held", "sharing", "ticks", "H", "done",
                     "crashed", "plateau_phase", "plateau_elapsed", "reply_value"):
            getattr(self, name)[mask] = 0
        self.queue[:, mask] = 0
        self.last_move[mask] = -self.lag
        self.p[mask] = 5
        self.safety[mask] = self.c.initial_safety
        self.safety_speed[mask] = self.target[mask] = 0.792
        self.next_change[mask] = self.c.safety_flat_until
        self.plateau_duration[mask] = 12 + 8 * _uniform(self.plateau_random, mask)
        self.reply_at[mask] = np.inf
        return self.observe()

    @staticmethod
    def _integral(phase, t, duration):
        r = t / duration
        smooth = duration * (r**3 - r**4 / 2)
        return np.select([phase == 0, phase == 1, phase == 2], [t, t - smooth, 0], default=smooth)

    def _safety(self, active):
        t = self.ticks * self.c.dt
        delta = np.where(active, np.clip(t - self.c.safety_flat_until, 0, self.c.dt), 0)
        change = (delta > 1e-10) & (t >= self.next_change)
        self.target[change] = 0.264 + _uniform(self.random, change)**1.5 * (1.584 - 0.264)
        self.next_change[change] = t[change] + 2 + 4 * _uniform(self.random, change)
        change_speed = -np.expm1(-delta)  # response time = 1 s
        rise = self.target * delta + (self.safety_speed - self.target) * change_speed
        self.safety_speed += (self.target - self.safety_speed) * change_speed
        if self.c.safety_plateaus:
            remaining, area = delta.copy(), np.zeros(self.n)
            while np.any(remaining > 1e-10):
                part = np.minimum(remaining, self.plateau_duration - self.plateau_elapsed)
                before = self._integral(self.plateau_phase, self.plateau_elapsed, self.plateau_duration)
                self.plateau_elapsed += part
                area += self._integral(self.plateau_phase, self.plateau_elapsed, self.plateau_duration) - before
                remaining = np.maximum(0, remaining - part)
                end = (delta > 1e-10) & (self.plateau_elapsed + 1e-10 >= self.plateau_duration)
                self.plateau_elapsed[end] = 0
                self.plateau_phase[end] = (self.plateau_phase[end] + 1) % 4
                self.plateau_duration[end] = 1
                rising, flat = end & (self.plateau_phase == 0), end & (self.plateau_phase == 2)
                self.plateau_duration[rising] = 12 + 8 * _uniform(self.plateau_random, rising)
                self.plateau_duration[flat] = 2 + 2 * _uniform(self.plateau_random, flat)
            scale = np.divide(area, delta, out=np.zeros_like(area), where=delta > 0)
            rise *= 1.2 * np.clip(scale, 0, 1)
        self.safety += rise

    def _bot(self):
        opponent = np.where(self.sharing[:, 0], self.x[:, 0], self.d[:, 0])
        target = np.maximum(self.safety, np.where(self.d[:, 1] < opponent, opponent + 1.5, 0))
        stopping = self.v[:, 1]**2 / (2 * self.c.deceleration * speed_limit(self.x[:, 1]))
        return self.x[:, 1] + stopping < target - np.where(self.held[:, 1], 0, 0.3)

    def step(self, actions, *, bot=True):
        actions = np.asarray(actions)
        if actions.shape != (self.n, 2) or actions.dtype.kind not in "iu" or np.any((actions < 0) | (actions > 3)):
            raise ValueError("actions must be integer (N, 2) values in [0, 3]")
        previous = self.cash.copy()
        live = ~self.done
        sharing = actions >= 2
        if bot:
            changed = live & (sharing[:, 0] != self.sharing[:, 0])
            self.reply_value[changed] = sharing[changed, 0]
            self.reply_at[changed] = np.where(self.sharing[changed, 1] == sharing[changed, 0], np.inf,
                                             self.ticks[changed] * self.c.dt + 2)
            self.sharing[live, 0] = sharing[live, 0]
        else:
            self.sharing[live] = sharing[live]
        for _ in range(self.repeat):
            live = ~self.done
            if not live.any():
                break
            running = live & (self.ticks < self.horizon)
            self.held[running, 0] = actions[running, 0] % 2 == 1
            control = self._bot() if bot else actions[:, 1] % 2 == 1
            self.held[running, 1] = control[running]
            old_rate = hazard(self.d, self.safety, self.c.max_hazard)
            old_profit = self.p.copy()
            limit = speed_limit(self.x)
            before = self.v.copy()
            after = np.where(self.held, np.minimum(limit, before + self.c.acceleration * limit * self.c.dt),
                             np.maximum(0, before - self.c.deceleration * limit * self.c.dt))
            elapsed = np.where(self.held, self.c.dt, np.minimum(self.c.dt, before / (self.c.deceleration * limit)))
            movement = (before + after) * elapsed / 2
            old_x = self.x.copy()
            self.x[live] += movement[live]
            self.v[live] = after[live]
            self.ticks[live] += 1
            self.last_move = np.where(live[:, None] & (self.x > old_x), self.ticks[:, None], self.last_move)
            self.queue[self.cursor] = self.x
            self.d[live] = self.queue[(self.cursor - self.lag) % len(self.queue), live]
            self.cursor = (self.cursor + 1) % len(self.queue)
            settling = live & (self.ticks >= self.horizon)
            self.held[settling] = False
            self.v[settling] = 0
            self._safety(live)
            rate = hazard(self.d, self.safety, self.c.max_hazard)
            self.H[live] += (old_rate[live] + rate[live]) * self.c.dt / 2
            self.p[live] = profit(self.d)[live]
            self.cash[live] = np.maximum(0, self.cash[live] + (old_profit[live] + self.p[live]) * self.c.dt * 7 / 365 / 2)
            crash = live & (self.H > 0) & (self.H >= self.threshold)
            finished = settling & (self.ticks >= self.last_move.max(axis=1) + self.lag)
            self.crashed |= crash
            self.done |= crash | finished
            self.cash[crash] = self.p[crash] = 0
            self.held[self.done] = False
            self.v[self.done] = 0
            if bot:
                reply = live & (self.ticks * self.c.dt + 1e-9 >= self.reply_at)
                self.sharing[reply, 1] = self.reply_value[reply]
                self.reply_at[reply] = np.inf
        # Telescopes to terminal cash, including the entire loss on catastrophe.
        return self.observe(), self.cash - previous, self.done.copy()

    def observe(self, player=0):
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        other = 1 - player
        visible = self.sharing[:, other] & ~self.done
        if not self.legacy:
            stopping = self.v[:, player]**2 / (2 * self.c.deceleration * speed_limit(self.x[:, player]))
            fields = [np.maximum(0, 1 - self.ticks * self.c.dt / self.c.duration), self.safety / 100,
                      (self.d[:, player] - self.safety) / 100, (self.d[:, other] - self.safety) / 100,
                      self.x[:, player] / 100, self.v[:, player] / 3.24, self.d[:, player] / 100, self.cash[:, player] / 20,
                      self.d[:, other] / 100, self.cash[:, other] / 20,
                      np.where(visible, self.x[:, other] / 100, 0), visible,
                      self.held[:, player], self.sharing[:, player], self.sharing[:, other], stopping / 100]
            return np.stack(fields, axis=-1).astype(np.float32)
        # Exact original observation layout for existing 23-input checkpoints.
        fields = [self.ticks * self.c.dt / self.c.duration, self.safety / 100, self.H,
                  (self.ticks < self.horizon) & ~self.done,
                  self.x[:, player] / 100, self.v[:, player] / 3.24, self.d[:, player] / 100,
                  self.cash[:, player] / 20, self.p[:, player] / 20,
                  self.held[:, player], self.sharing[:, player],
                  self.d[:, other] / 100, self.cash[:, other] / 20, self.p[:, other] / 20,
                  self.sharing[:, other], np.where(visible, self.x[:, other] / 100, 0), visible,
                  hazard(self.d, self.safety, self.c.max_hazard)]
        return np.stack(fields, axis=-1).astype(np.float32)
