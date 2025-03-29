import os
import time
import errno
import datetime
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """
    Tracks a series of values and provides access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize the total and count across all processes."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg,
            global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """Gathers arbitrary picklable data from all processes."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """Reduce dictionary values across all processes by summing or averaging."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.inference_mode():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys], dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(keys, values)}


class MetricLogger:
    """Utility class for logging training metrics."""

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join([f"{name}: {meter}" for name, meter in self.meters.items()])

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i, start_time, end = 0, time.time(), time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = f":{len(str(len(iterable)))}d"

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header or "",
                f"[{{i{space_fmt}}}/{{len_iterable}}]",
                "eta: {eta}", "{meters}",
                "time: {time}", "data: {data}",
                "max mem: {memory:.0f}"
            ])
        else:
            log_msg = self.delimiter.join([
                header or "",
                f"[{{i{space_fmt}}}/{{len_iterable}}]",
                "eta: {eta}", "{meters}",
                "time: {time}", "data: {data}"
            ])

        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print_kwargs = {
                    "i": i,
                    "len_iterable": len(iterable),
                    "eta": eta_string,
                    "meters": str(self),
                    "time": str(iter_time),
                    "data": str(data_time),
                }

                if torch.cuda.is_available():
                    print_kwargs["memory"] = torch.cuda.max_memory_allocated() / MB
                print(log_msg.format(**print_kwargs))

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        print(f"{header or ''} Total time: {str(datetime.timedelta(seconds=int(total_time)))} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    """Collate function to group items into batches."""
    return tuple(zip(*batch))


def mkdir(path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """Disable printing on non-master processes."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """Initialize distributed training mode from environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = "nccl"
    torch.cuda.set_device(args.gpu)

    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)
