import time
import torch
__all__ = ["pb4dl"]


def timeToStr(num) -> str:
    def day(x):
        return int(x / (60 * 60 * 24))

    def hour(x):
        return int(x / (60 * 60))

    def minute(x):
        return int(x / 60)

    num = int(num)
    if num > 60 * 60 * 24:
        days = day(num)
        hours = hour(num - days * 60 * 60 * 24)
        minutes = minute(num - days * 60 * 60 * 24 - hours * 60 * 60)
        seconds = num - days * 60 * 60 * 24 - hours * 60 * 60 - minutes * 60
        return f'{days}d {hours}h {minutes}m {seconds}s'
    if num > 60 * 60:
        hours = hour(num)
        minutes = minute(num - hours * 60 * 60)
        seconds = num - hours * 60 * 60 - minutes * 60
        return f'{hours}h {minutes}m {seconds}s'
    if num > 60:
        minutes = minute(num)
        seconds = num - minutes * 60
        return f'{minutes}m {seconds}s'
    else:
        return f'{num}s'


def progressPrint(numNow: int, numAll: int, needTime, word: str, device: str):
    if numNow > numAll:
        raise ValueError("Maximum number exceeded")
    percent = numNow / numAll
    part1 = int(percent * 50)
    part2 = 50 - part1
    status = "✔ done!\n" if numNow == numAll else device

    print(f"\r {word} [{part1 * '-'}>{part2 * ' '}] {percent * 100:.2f}% | {numNow}/{numAll} | {needTime} | {status}",
          end='')


class pb4dl:
    epoch = {}

    def __init__(self, dataloader):
        assert dataloader.__class__.__name__ == "DataLoader", 'the input is not DataLoader class'
        if dataloader not in pb4dl.epoch:
            pb4dl.epoch[dataloader] = 0
        self.dataloader = dataloader
        self.dataloaderIter = iter(dataloader)
        self.size = len(dataloader)
        self.num = 0
        self.startTime = time.time()
        self.cuda = torch.cuda.is_available()
        pb4dl.epoch[dataloader] += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.num != 0:
            differenceTime = time.time() - self.startTime
            needTime = differenceTime * self.size / self.num - differenceTime
            needTime = timeToStr(needTime)
            word = f"epoch{pb4dl.epoch[self.dataloader]}"
            if self.cuda:
                device_id = torch.cuda.current_device()
                device = f"cuda{device_id} : {torch.cuda.memory_reserved(device_id)/(1024 ** 3)}"
            else:
                device = "cpu"
            progressPrint(self.num, self.size, needTime, word,device)
        self.num += 1
        return next(self.dataloaderIter)
