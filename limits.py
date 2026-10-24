# 5, 6, yt

import hashlib
import json
import random
import time


class TransientError(Exception): pass
class PermanentError(Exception): pass


def retry_with_backoff(fn, max_attempts=4, base_delay=0.5, max_delay=8.0,
                       sleep=time.sleep, jitter=True):
    delays = []
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), delays
        except TransientError:
            if attempt == max_attempts:
                raise  
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            if jitter:
                delay = random.uniform(0, delay)
            delays.append(delay)
            sleep(delay)



class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_tokens, max_seconds, clock=time.monotonic):
        self.max_tokens, self.max_seconds, self.clock = max_tokens, max_seconds, clock
        self.used_tokens = 0
        self.started = clock()

    def charge(self, tokens):
        self.used_tokens += tokens
        if self.used_tokens > self.max_tokens:
            raise BudgetExceeded(f"token budget of {self.max_tokens} exceeded")
        if self.clock() - self.started > self.max_seconds:
            raise BudgetExceeded(f"time budget of {self.max_seconds}s exceeded")



class TokenBucket:
    """Allows `capacity` actions in a burst, then refills `rate` tokens per second."""
    def __init__(self, capacity, rate, clock=time.monotonic):
        self.capacity, self.rate, self.clock = capacity, rate, clock
        self.tokens = float(capacity)
        self.last = clock()

    def allow(self, cost=1):
        now_ = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now_ - self.last) * self.rate)
        self.last = now_
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class IdempotentExecutor:
    def __init__(self):
        self.done = {} 

    @staticmethod
    def key_for(action, payload):
        raw = json.dumps({"action": action, "payload": payload}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def run(self, action, payload, fn):
        key = self.key_for(action, payload)
        if key in self.done:
            return self.done[key], "skipped (already done)"
        result = fn(**payload)
        self.done[key] = result
        return result, "executed"