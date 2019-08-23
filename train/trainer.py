import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


def reduce_distributed_output(output, nb_gpus):
    if nb_gpus <= 1:
        return output

    # when using DP, we get one output per gpu
    # average outputs and return
    if type(output) is torch.Tensor:
        return output.mean()

    for k, v in output.items():
        # recurse on nested dics
        if isinstance(output[k], dict):
            output[k] = reduce_distributed_output(output[k], nb_gpus)

        # reduce only metrics that have the same nb of gpus
        elif output[k].size(0) == nb_gpus:
            reduced = torch.mean(output[k])
            output[k] = reduced
    return output


class Trainer(object):
    def __init__(self,
                 output_dir,
                 accumulation_grad_batches: int = 1,
                 gpus=None):
        self.model = None
        self.accumulation_grad_batches = accumulation_grad_batches
        self.lr_schedulers = []
        self.optimizers = []

        self.summary_writer = SummaryWriter(output_dir)

    def train(self, model, epoch, validate_steps):
        pass

    def evaluate(self, model):
        pass

    def write_metrics(self, model, step, reset=False):
        metrics = model.get_metric(reset)
        for k, v in metrics.items():
            self.summary_writer.add_scalar(k, v, global_step=step)

    @classmethod
    def from_config(cls, config):
        return cls()
