import logging
import math
import os
import time
from collections import OrderedDict

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from skimage import io

from models.model import Baseline
from train import train, test, visualize_testloader
from utils import ISPRS_dataset, convert_to_color, fix_random_seed

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def _load_model_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']

    model_state_keys = list(model.state_dict().keys())
    ckpt_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    if ckpt_keys and model_state_keys:
        ckpt_has_module = ckpt_keys[0].startswith('module.')
        model_has_module = model_state_keys[0].startswith('module.')
        if ckpt_has_module and not model_has_module:
            checkpoint = OrderedDict((k.replace('module.', '', 1), v) for k, v in checkpoint.items())
        elif not ckpt_has_module and model_has_module:
            checkpoint = OrderedDict((f'module.{k}', v) for k, v in checkpoint.items())

    load_result = model.load_state_dict(checkpoint, strict=False)
    if getattr(load_result, 'missing_keys', None):
        logger.warning("Missing keys when loading checkpoint %s: %d | sample: %s", checkpoint_path,
                       len(load_result.missing_keys), load_result.missing_keys[:20])
    if getattr(load_result, 'unexpected_keys', None):
        logger.warning("Unexpected keys when loading checkpoint %s: %d | sample: %s", checkpoint_path,
                       len(load_result.unexpected_keys), load_result.unexpected_keys[:20])


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    fix_random_seed(cfg.seed)

    logger.info("Loaded configuration:")
    logger.info("--- training configuration:")
    logger.info("\n%s", OmegaConf.to_yaml(cfg.training, resolve=True))
    logger.info("--- model configuration:")
    logger.info("\n%s", OmegaConf.to_yaml(cfg.model, resolve=True))

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.cuda_visible_devices))
    logger.info("Using GPUs: %s %d", os.environ["CUDA_VISIBLE_DEVICES"], len(cfg.cuda_visible_devices))
    logger.info("Using model: %s", cfg.training_model)
    logger.info("Using dataset: %s", cfg.training_dataset)

    cfg.training.batch_size = cfg.training.batch_size * len(cfg.cuda_visible_devices)
    cfg.training.learning_rate = cfg.training.learning_rate * len(cfg.cuda_visible_devices)

    logger.info("Real Training Configuration:")
    logger.info("batch size: %d", cfg.training.batch_size)
    logger.info("learning rate: %f", cfg.training.learning_rate)

    dataset_cfg = cfg.dataset.datasets[cfg.training_dataset]
    N_CLASSES = dataset_cfg.n_classes
    WEIGHTS = torch.ones(N_CLASSES)

    train_ids = dataset_cfg.train_ids
    test_ids = dataset_cfg.test_ids

    model_cfg = cfg.model
    if cfg.training_dataset == 'Potsdam':
        model = Baseline(cfg=model_cfg, num_classes=N_CLASSES, in_chans=[4, 1])
    elif cfg.training_dataset == 'Vaihingen':
        model = Baseline(cfg=model_cfg, num_classes=N_CLASSES, in_chans=[3, 1])

    model = model.cuda()
    model = nn.DataParallel(model)

    robust_kd_cfg = cfg.training.get('robust_kd', None)
    robust_kd_enabled = bool(robust_kd_cfg.get('enabled', False)) if robust_kd_cfg is not None else False
    teacher_model = None
    if robust_kd_enabled:
        logger.info("Robust KD enabled")
        if cfg.training_dataset == 'Potsdam':
            teacher_model = Baseline(cfg=model_cfg, num_classes=N_CLASSES, in_chans=[4, 1])
        elif cfg.training_dataset == 'Vaihingen':
            teacher_model = Baseline(cfg=model_cfg, num_classes=N_CLASSES, in_chans=[3, 1])
        else:
            teacher_model = Baseline(cfg=model_cfg, num_classes=N_CLASSES, in_chans=[3, 1])
        teacher_model = teacher_model.cuda()
        teacher_model = nn.DataParallel(teacher_model)

        if robust_kd_cfg.get('teacher_checkpoint', ''):
            _load_model_state(teacher_model, robust_kd_cfg.teacher_checkpoint)
            logger.info("Loaded teacher checkpoint from %s", robust_kd_cfg.teacher_checkpoint)
        else:
            teacher_model.load_state_dict(model.state_dict(), strict=True)
            logger.info("Initialized teacher from current student weights")
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()

    total_params = sum(param.nelement() for param in model.parameters())
    logger.info('All Params: %d', total_params)

    logger.info("training : %s", train_ids)
    logger.info("testing : %s", test_ids)

    if cfg.training_dataset == 'WHU':
        train_dataset = WHUDataset(
            data_dir=os.path.join(cfg.folder, 'OPT_SAR_WHU_crops'),
            split='train',
            color_map={k: tuple(v) for k, v in dataset_cfg.palette.items()}
        )
        test_dataset = WHUDataset(
            data_dir=os.path.join(cfg.folder, 'OPT_SAR_WHU_crops'),
            split='test',
            color_map={k: tuple(v) for k, v in dataset_cfg.palette.items()}
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                                   num_workers=cfg.training.num_workers, pin_memory=True,
                                                   worker_init_fn=lambda worker_id: fix_random_seed(
                                                       cfg.seed + worker_id))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                                                  num_workers=cfg.training.num_workers, pin_memory=True,
                                                  worker_init_fn=lambda worker_id: fix_random_seed(
                                                      cfg.seed + worker_id))
    elif cfg.training_dataset == 'YESeg':
        train_dataset = YESegDataset(
            data_dir=os.path.join(cfg.folder, 'YESeg-OPT-SAR-split'),
            split='train',
            color_map={k: tuple(v) for k, v in dataset_cfg.palette.items()}
        )
        test_dataset = YESegDataset(
            data_dir=os.path.join(cfg.folder, 'YESeg-OPT-SAR-split'),
            split='test',
            color_map={k: tuple(v) for k, v in dataset_cfg.palette.items()}
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                                   num_workers=cfg.training.num_workers, pin_memory=True,
                                                   worker_init_fn=lambda worker_id: fix_random_seed(
                                                       cfg.seed + worker_id))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                                                  num_workers=cfg.training.num_workers, pin_memory=True,
                                                  worker_init_fn=lambda worker_id: fix_random_seed(
                                                      cfg.seed + worker_id))
    else:
        train_dataset = ISPRS_dataset(train_ids, dataset_cfg, cfg.training.window_size, cache=cfg.training.cache,
                                      augmentation=cfg.training.augmentation)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                                   num_workers=cfg.training.num_workers, pin_memory=True,
                                                   worker_init_fn=lambda worker_id: fix_random_seed(
                                                       cfg.seed + worker_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                                  weight_decay=0.01)
    niters_per_epoch = math.ceil(dataset_cfg.num_train_imgs / cfg.training.batch_size)
    total_iteration = cfg.training.epochs * niters_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_iteration)

    if cfg.training_dataset == 'Vaihingen':
        results_dir = os.path.join(os.getcwd(), 'results_{}_vaihingen'.format(cfg.training_model))
    elif cfg.training_dataset == 'Potsdam':
        results_dir = os.path.join(os.getcwd(), 'results_{}_potsdam'.format(cfg.training_model))
    elif cfg.training_dataset == 'WHU':
        results_dir = os.path.join(os.getcwd(), 'results_{}_whu'.format(cfg.training_model))
    elif cfg.training_dataset == 'YESeg':
        results_dir = os.path.join(os.getcwd(), 'results_{}_yeseg'.format(cfg.training_model))
    else:
        raise ValueError("Unsupported dataset: " + cfg.training_dataset)
    os.makedirs(results_dir, exist_ok=True)
    logger.info("Experiment output directory: %s", os.getcwd())

    start_train = time.time()
    logger.info('Start training...')
    if cfg.training_dataset == 'WHU' or cfg.training_dataset == 'YESeg':
        train(dataset_cfg, cfg.training, model, optimizer, scheduler, train_loader, WEIGHTS, results_dir,
              test_loader=test_loader, teacher_model=teacher_model)
    else:
        train(dataset_cfg, cfg.training, model, optimizer, scheduler, train_loader, WEIGHTS, results_dir,
              teacher_model=teacher_model)
    end_train = time.time()
    logger.info('Training time: {:.2f} hours'.format((end_train - start_train) / 3600))
    logger.info("")

    start_test = time.time()
    logger.info('Start testing...')
    palette = {k: tuple(v) for k, v in dataset_cfg.palette.items()}
    # test
    if cfg.training_dataset == 'Vaihingen':
        best_model_path = os.path.join(results_dir, 'final_model_vaihingen')
        model.load_state_dict(torch.load(best_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True)
        logger.info('Final model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p, palette)
            io.imsave(os.path.join(results_dir, 'best_model_inference_{}_tile_{}.png'.format(cfg.training_model, id_)),
                      img)

    elif cfg.training_dataset == 'Potsdam':
        best_model_path = os.path.join(results_dir, 'final_model_potsdam')
        model.load_state_dict(torch.load(best_model_path, weights_only=True), strict=False)
        model.eval()
        results, all_preds, all_gts = test(dataset_cfg, cfg.training, model, dataset_cfg.test_ids, all=True)
        logger.info('Final model result:')
        logger.info('    Kappa: %s', results["Kappa"])
        logger.info('    OA: %s', results["OA"])
        logger.info('    F1: %s', results["F1"])
        logger.info('    MIoU: %s', results["MIoU"])
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p, palette)
            io.imsave(os.path.join(results_dir, 'best_model_inference_{}_tile_{}.png'.format(cfg.training_model, id_)),
                      img)

    end_test = time.time()
    logger.info('Testing time: {:.2f} hours'.format((end_test - start_test) / 3600))


if __name__ == '__main__':
    main()
