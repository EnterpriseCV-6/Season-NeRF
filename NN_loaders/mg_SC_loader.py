import torch as t
import torch.utils.data as tdata

def setup_SC_loader(loader_size, bounds = t.tensor([[-1.,1], [-1,1], [-1,1]])):
    return SC_loader(loader_size, bounds)

class SC_loader(tdata.Dataset):
    def __init__(self, loader_size, bounds):
        self.bounds = bounds
        self.dx = bounds[0,1] - bounds[0,0]
        self.x = bounds[0,0]
        self.dy = bounds[1,1] - bounds[1,0]
        self.y = bounds[1,0]
        self.size = loader_size


    def __getitem__(self, item):
        rands = t.rand(4)
        tops = t.tensor([rands[0] * self.dx + self.x, rands[1] * self.dy + self.y, self.bounds[2, 1]])
        bots = t.tensor([rands[2] * self.dx + self.x, rands[3] * self.dy + self.y, self.bounds[2, 0]])
        return tops, bots


    def __len__(self):
        return self.size
