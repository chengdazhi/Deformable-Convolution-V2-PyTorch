from mmcv.runner import Runner
from mmcv.runner.utils import obj_from_dict
import torch

class MultiLRRunner(Runner):
    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
             optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
             type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            lr_mult_dict = {1.:[]}
            base_lr = optimizer['lr']
            for m in self.model.modules():
                if len([_ for _ in m.children()]) > 0:
                    children_params = []
                    for child in m.children():
                        children_params += [id(_) for _ in child.parameters()]
                    for _param in [_ for _ in m.parameters()]:
                        if id(_param) not in children_params:
                            lr_mult_dict[1.].append(_param)
                    continue # not simplest node
                if hasattr(m, 'lr_mult'):
                    if m.lr_mult not in lr_mult_dict.keys():
                        cur_lr_params = []
                        lr_mult_dict[m.lr_mult] = cur_lr_params
                    else:
                        cur_lr_params = lr_mult_dict[m.lr_mult]
                    for _param in m.parameters():
                        cur_lr_params.append(_param)
                else:
                    for _param in m.parameters():
                        lr_mult_dict[1.].append(_param)

            params_dict = []
            for lr_mult in lr_mult_dict.keys():
                params_dict.append({'params': lr_mult_dict[lr_mult], 'lr': base_lr * lr_mult})

            optimizer = obj_from_dict(optimizer, torch.optim, dict(params=params_dict))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer