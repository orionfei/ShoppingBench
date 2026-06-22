import logging

import torch.distributed as dist


class LocalLogger:
    def __init__(self, print_to_console: bool = True):
        self.print_to_console = print_to_console

    def log(self, data, step):
        if not self.print_to_console:
            return
        items = ", ".join(f"{key}={value}" for key, value in sorted(data.items()))
        print(f"[step {step}] {items}")


class DecoratorLoggerBase:
    def __init__(
        self,
        role: str,
        logger: logging.Logger = None,
        level: int = logging.DEBUG,
        rank: int = 0,
        log_only_rank_0: bool = True,
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0

    def should_log(self) -> bool:
        if not self.log_only_rank_0:
            return True
        if not dist.is_available() or not dist.is_initialized():
            return self.rank == 0
        return dist.get_rank() == 0

    def logging_function(self, message: str) -> None:
        if not self.should_log():
            return
        message = f"[{self.role}] {message}"
        if self.logger is None:
            print(message)
        else:
            self.logger.log(self.level, message)


def _current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def log_with_rank(
    message: str,
    rank: int = 0,
    logger: logging.Logger = None,
    level: int = logging.INFO,
    log_only_rank_0: bool = False,
) -> None:
    current_rank = _current_rank()
    should_log = current_rank == 0 if log_only_rank_0 else rank is None or current_rank == rank
    if should_log:
        if logger is None:
            print(message)
        else:
            logger.log(level, message)


def print_rank_0(*args, **kwargs) -> None:
    if _current_rank() == 0:
        print(*args, **kwargs)
