# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    
    if cfg.MODEL.CENTER_ON:
        params_center = []
#         print("model info")
#         print(model.named_parameters())
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            if 'da_heads' not in str(key):
                continue
#             print(key,value.requires_grad)
            lr_center = cfg.SOLVER_CENTER.BASE_LR
            weight_decay_center = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr_center = cfg.SOLVER_CENTER.BASE_LR * cfg.SOLVER_CENTER.BIAS_LR_FACTOR
                weight_decay_center = cfg.SOLVER_CENTER.WEIGHT_DECAY_BIAS
            params_center += [{"params": [value], "lr": lr_center, "weight_decay": weight_decay_center}]
            
        optimizer_center = torch.optim.SGD(params_center, lr_center, momentum=cfg.SOLVER_CENTER.MOMENTUM)
        
        return [optimizer,optimizer_center]
    else:
        return [optimizer,None]
#     return optimizer


def make_lr_scheduler(cfg, optimizer):
    
    lr1 = WarmupMultiStepLR(
        optimizer[0],
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    if optimizer[1]:
        lr2 = WarmupMultiStepLR(
            optimizer[1],
            cfg.SOLVER_CENTER.STEPS,
            cfg.SOLVER_CENTER.GAMMA,
            warmup_factor=cfg.SOLVER_CENTER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER_CENTER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER_CENTER.WARMUP_METHOD,
        )
    else:
        lr2 = None
    return [lr1,lr2]
