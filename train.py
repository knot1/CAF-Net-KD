import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import io

from utils import format_string, convert_from_color, count_sliding_window, grouper, sliding_window, CrossEntropy2d, dice_loss, \
    metrics, convert_to_color

# from loss.uncertainty import uncertainty_loss

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def _get_cfg_value(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _kl_divergence_logits(student_logits, teacher_logits, temperature=1.0):
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temperature ** 2)


def _degrade_input_batch(rgb, dsm, robust_kd_cfg):
    modes = _get_cfg_value(robust_kd_cfg, 'modes', ['rgb_noise', 'rgb_missing', 'dsm_missing', 'dsm_hole', 'resolution_down'])
    mode = random.choice(list(modes))

    rgb_degraded = rgb.clone()
    dsm_degraded = dsm.clone()

    if mode == 'rgb_noise':
        noise_std = float(_get_cfg_value(robust_kd_cfg, 'noise_std', 0.1))
        rgb_degraded = rgb_degraded + torch.randn_like(rgb_degraded) * noise_std
        rgb_degraded = torch.clamp(rgb_degraded, 0.0, 1.0)

    elif mode == 'rgb_missing':
        rgb_degraded = torch.zeros_like(rgb_degraded)

    elif mode == 'dsm_missing':
        dsm_degraded = torch.zeros_like(dsm_degraded)

    elif mode == 'dsm_hole':
        dsm_hole_ratio = float(_get_cfg_value(robust_kd_cfg, 'dsm_hole_ratio', 0.1))
        dsm_hole_ratio = min(max(dsm_hole_ratio, 0.0), 1.0)
        hole_mask = torch.rand_like(dsm_degraded) < dsm_hole_ratio
        dsm_degraded[hole_mask] = 0.0

    elif mode == 'resolution_down':
        resolution_scale = float(_get_cfg_value(robust_kd_cfg, 'resolution_scale', 0.5))
        resolution_scale = min(max(resolution_scale, 0.1), 1.0)
        h, w = rgb_degraded.shape[-2:]
        down_h = max(1, int(h * resolution_scale))
        down_w = max(1, int(w * resolution_scale))
        rgb_degraded = F.interpolate(rgb_degraded, size=(down_h, down_w), mode='bilinear', align_corners=False)
        rgb_degraded = F.interpolate(rgb_degraded, size=(h, w), mode='bilinear', align_corners=False)

    return rgb_degraded, dsm_degraded, mode


@torch.no_grad()
def _update_ema_teacher(student_model, teacher_model, momentum):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=(1.0 - momentum))
    for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
        teacher_buffer.data.copy_(student_buffer.data)


def test(dataset_cfg, training_cfg, model, test_ids, all=False, test_loader=None):
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen':
        stride = dataset_cfg.stride_size
    batch_size = training_cfg.batch_size
    window_size = tuple(training_cfg.window_size)
    N_CLASSES = dataset_cfg.n_classes
    # Use the network on the test set
    if dataset_cfg.name == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32')
                       for id in
                       test_ids)
    elif dataset_cfg.name == 'Vaihingen':
        test_images = (1 / 255 * np.asarray(io.imread(dataset_cfg.data_folder.format(id)), dtype='float32') for id in
                       test_ids)

    if dataset_cfg.name == 'Potsdam':
        # dif_ids = [format_string(id) for id in test_ids]
        dif_ids = [id for id in test_ids]
        test_dsms = (np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32') for id in dif_ids)
    else:
        test_dsms = (np.asarray(io.imread(dataset_cfg.dsm_folder.format(id)), dtype='float32') for id in test_ids)

    invert_palette = {tuple(v): k for k, v in dataset_cfg.palette.items()}
    test_labels = (convert_from_color(io.imread(dataset_cfg.label_folder.format(id)), invert_palette) for id in
                   test_ids)
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen':
        eroded_labels = (convert_from_color(io.imread(dataset_cfg.eroded_folder.format(id)), invert_palette) for id in
                         test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    if dataset_cfg.name == 'Potsdam' or dataset_cfg.name == 'Vaihingen':
        with torch.no_grad():
            for img, dsm, gt, gt_e in zip(test_images, test_dsms, test_labels, eroded_labels):
                pred = np.zeros(img.shape[:2] + (N_CLASSES,))

                total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
                for i, coords in enumerate(
                        grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                    # Build the tensor
                    image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = torch.from_numpy(image_patches).cuda()

                    min = np.min(dsm)
                    max = np.max(dsm)
                    dsm = (dsm - min) / (max - min)
                    dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                    dsm_patches = np.asarray(dsm_patches)
                    dsm_patches = torch.from_numpy(dsm_patches).cuda()

                    # Do the inference
                    outs, _, _ = model(image_patches, dsm_patches)
                    outs = outs.data.cpu().numpy()

                    # Fill in the results array
                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1, 2, 0))
                        pred[x:x + w, y:y + h] += out
                    del (outs)

                pred = np.argmax(pred, axis=-1)
                all_preds.append(pred)
                all_gts.append(gt_e)
                # clear_output()

        results = metrics(np.concatenate([p.ravel() for p in all_preds]),
                          np.concatenate([p.ravel() for p in all_gts]).ravel(), dataset_cfg.labels,
                          dataset_cfg.n_classes)

        if all:
            return results, all_preds, all_gts
        else:
            return results



        results = metrics(np.concatenate([p.ravel() for p in all_preds]),
                          np.concatenate([p.ravel() for p in all_gts]).ravel(), dataset_cfg.labels,
                          dataset_cfg.n_classes)

        if all:
            return results, all_preds, all_gts
        else:
            return results


def train(dataset_cfg, training_cfg, model, optimizer, scheduler, train_loader, weights, results_dir, test_loader=None,
          teacher_model=None):
    weights = weights.cuda()
    epochs = training_cfg.epochs
    save_epoch = training_cfg.save_epoch
    robust_kd_cfg = _get_cfg_value(training_cfg, 'robust_kd', None)
    robust_kd_enabled = bool(_get_cfg_value(robust_kd_cfg, 'enabled', False))
    kd_weight = float(_get_cfg_value(robust_kd_cfg, 'kd_weight', 1.0))
    consistency_weight = float(_get_cfg_value(robust_kd_cfg, 'consistency_weight', 0.0))
    kd_temperature = float(_get_cfg_value(robust_kd_cfg, 'temperature', 1.0))
    consistency_temperature = float(_get_cfg_value(robust_kd_cfg, 'consistency_temperature', 1.0))
    teacher_use_ema = bool(_get_cfg_value(robust_kd_cfg, 'teacher_use_ema', False))
    ema_momentum = float(_get_cfg_value(robust_kd_cfg, 'ema_momentum', 0.999))

    if robust_kd_enabled and teacher_model is None:
        logger.warning('robust_kd.enabled=True but teacher model is None; falling back to no-grad clean forward of student.')
    if robust_kd_enabled and teacher_model is not None:
        teacher_model.eval()

    history = {
        'round': [],
        'train_loss': [],
        'Kappa': [],
        'OA_total': [],
        'MIoU_mean': [],
        'F1_mean': []
    }

    for label in dataset_cfg.labels:
        history[f'OA_{label}'] = []
        history[f'MIoU_{label}'] = []
        history[f'F1_{label}'] = []

    MIoU_best = 0.0
    epoch_best = -1
    for epoch in range(1, epochs + 1):
        logger.info('Train (epoch {}/{})'.format(epoch, epochs))
        model.train()
        batch_losses = []
        total_iter = len(train_loader)
        print_interval = max(1, total_iter // 10)
        for batch_idx, (opt, dsm, target) in enumerate(train_loader):
            opt, dsm, target = opt.cuda(), dsm.cuda(), target.cuda()
            optimizer.zero_grad()

            if robust_kd_enabled:
                opt_degraded, dsm_degraded, _ = _degrade_input_batch(opt, dsm, robust_kd_cfg)
                student_logits_deg, L_cons, low_L_cons = model(opt_degraded, dsm_degraded)
                loss_ce = CrossEntropy2d(student_logits_deg, target, weight=weights)
                loss_dice = dice_loss(student_logits_deg, target)
                loss = loss_ce + (L_cons * training_cfg.alpha) - (low_L_cons * training_cfg.beta) + (
                            loss_dice * training_cfg.gamma)

                with torch.no_grad():
                    if teacher_model is not None:
                        teacher_logits_clean, _, _ = teacher_model(opt, dsm)
                    else:
                        teacher_logits_clean, _, _ = model(opt, dsm)
                loss_kd = _kl_divergence_logits(student_logits_deg, teacher_logits_clean, temperature=kd_temperature)
                loss = loss + kd_weight * loss_kd

                if consistency_weight > 0:
                    student_logits_clean, _, _ = model(opt, dsm)
                    loss_consistency = _kl_divergence_logits(student_logits_deg, student_logits_clean.detach(),
                                                             temperature=consistency_temperature)
                    loss = loss + consistency_weight * loss_consistency
            else:
                output, L_cons, low_L_cons = model(opt, dsm)
                loss_ce = CrossEntropy2d(output, target, weight=weights)
                loss_dice = dice_loss(output, target)
                loss = loss_ce + (L_cons * training_cfg.alpha) - (low_L_cons * training_cfg.beta) + (
                            loss_dice * training_cfg.gamma)
            loss.backward()
            optimizer.step()
            if robust_kd_enabled and teacher_model is not None and teacher_use_ema:
                _update_ema_teacher(model, teacher_model, momentum=ema_momentum)

            if scheduler is not None:
                scheduler.step()

            batch_losses.append(loss.item())
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_iter:
                print(f"Iter {batch_idx+1}/{total_iter} | Loss: {loss.item():.4f}")
            del (opt, target, loss)

        epoch_loss = np.mean(batch_losses)

        if epoch % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            model.eval()
            
            results_val = test(dataset_cfg, training_cfg, model, dataset_cfg.test_ids, all=False)
            model.train()

            MIoU = results_val['MIoU']['mean']

            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(results_val['Kappa'])

            history['OA_total'].append(results_val['OA']['total'])
            for i in dataset_cfg.labels:
                history['OA_{}'.format(i)].append(results_val['OA'][i])

            history['MIoU_mean'].append(results_val['MIoU']['mean'])
            for i in dataset_cfg.labels:
                history['MIoU_{}'.format(i)].append(results_val['MIoU'][i])

            history['F1_mean'].append(results_val['F1']['mean'])
            for i in dataset_cfg.labels:
                history['F1_{}'.format(i)].append(results_val['F1'][i])

            if MIoU > MIoU_best:
                if dataset_cfg.name == 'Vaihingen':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_vaihingen'))
                elif dataset_cfg.name == 'Potsdam':
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_potsdam'))

                MIoU_best = MIoU
                epoch_best = epoch

            logger.info('    Training Loss: {}'.format(epoch_loss))
            logger.info('    Kappa: {}'.format(results_val["Kappa"]))
            logger.info('    OA: {}'.format(results_val["OA"]))
            logger.info('    F1: {}'.format(results_val["F1"]))
            logger.info('    MIoU: {}'.format(results_val["MIoU"]))
            logger.info("")
        else:
            history['round'].append(epoch)
            history['train_loss'].append(epoch_loss)
            history['Kappa'].append(0.0)

            history['OA_total'].append(0.0)
            for i in dataset_cfg.labels:
                history['OA_{}'.format(i)].append(0.0)

            history['MIoU_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history['MIoU_{}'.format(i)].append(0.0)

            history['F1_mean'].append(0.0)
            for i in dataset_cfg.labels:
                history['F1_{}'.format(i)].append(0.0)

            logger.info('    Training Loss: {}'.format(epoch_loss))

    logger.info('Best epoch {}, MIoU best: {}'.format(epoch_best, MIoU_best))

    df = pd.DataFrame(history)
    if dataset_cfg.name == 'Vaihingen':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_vaihingen'))
    elif dataset_cfg.name == 'Potsdam':
        df.to_csv(os.path.join(results_dir, 'history.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(results_dir, 'final_model_potsdam'))

    logger.info('End of training !')


def visualize_testloader(model, test_loader, palette, save_root):
    # os.makedirs(save_root, exist_ok=True)
    model.cuda()
    model.eval()
    tile_idx = 0
    with torch.no_grad():
        for img, dsm, _ in test_loader:
            img, dsm = img.cuda(), dsm.cuda()
            pred, _, _ = model(img, dsm)
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            for i in range(pred.shape[0]):
                color_pred = convert_to_color(pred[i], palette)
                io.imsave(os.path.join(save_root, f"tile_{tile_idx}.png"),
                          color_pred, check_contrast=False)
                tile_idx += 1
