import math
import sys
import time

import torch

import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from easydict import EasyDict
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy import stats



def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)  #得到loss字典

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr



def gen_area_dict(label_list):
    #{1: 'Table', 2: 'Text', 3: 'Title', 4: 'Image'}
    label_dict = label_list[0]
    text_area = 0
    #targets的
    if "area" in label_dict.keys():
        class_labels = label_dict["labels"]
        area = label_dict["area"]
        text_area = 0
        total_area = 0
        for class_label,class_area in zip(class_labels,area):
            total_area += class_area
            if class_label == 2 or class_label == 3:
                text_area += class_area
    else:
        #outputs的
        class_labels = label_dict["labels"]
        masks =label_dict["masks"]
        total_area = 0
        for class_label,mask in zip(class_labels,masks):
            mask = mask.detach().numpy()
            mask = mask.squeeze()
            class_area = np.sum(mask)
            class_area = float(class_area)
            total_area += class_area
            if class_label == 2 or class_label == 3:
                text_area += class_area

    return text_area,total_area

def compute_ptrate(image_id,text_area,total_area):
    rate_dict = {}
    try:
        text_rate = text_area /total_area
        rate_dict[image_id] = text_rate
    except ZeroDivisionError:
        print("**********************************************", image_id)
        print("**********************************************", text_area)
        print("**********************************************",total_area)
        rate_dict[image_id] = 0
    return rate_dict

def compt_score(rate_dict):
    score_dict={}
    image_id = list(rate_dict.keys())[0]
    rate = list(rate_dict.values())[0]

    if rate >= 0.9 and rate <= 1:
        temp_score  = 1
    elif (rate >= 0 and rate < 0.1) or (rate >= 0.8 and rate < 0.9):
        temp_score = 3
    elif (rate >= 0.1 and rate < 0.2) or (rate >= 0.7 and rate < 0.8):
        temp_score = 5
    elif (rate >= 0.2 and rate < 0.3) or (rate >= 0.6 and rate < 0.7):
        temp_score = 7
    elif rate >= 0.3 and rate < 0.4:
        temp_score = 15
    elif rate >= 0.4 and rate < 0.5:
        temp_score = 12
    elif rate >= 0.5 and rate < 0.6:
        temp_score = 9
    else:
        temp_score = 0

    score_dict[image_id]=temp_score
    return score_dict





@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")
    true_score_list = []
    pred_score_list = []
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)   #image(3,720,1280)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        #计算分散
        result = EasyDict()
        label_dict = targets[0]
        if "area" in label_dict.keys():
            image_id = label_dict["image_id"]
        #计算分散
        evg_pre_textarea,evg_pre_totalarea = torch.tensor(gen_area_dict(outputs))
        evg_true_textarea,evg_true_totalarea = gen_area_dict(targets)

        ture_textrate_dict = compute_ptrate(image_id,evg_true_textarea,evg_true_totalarea )
        pre_textrate_dict = compute_ptrate(image_id,evg_pre_textarea,evg_pre_totalarea)

        evg_ture_score_dict = compt_score(ture_textrate_dict)
        evg_pre_score_dict = compt_score(pre_textrate_dict)

        evg_ture_score = list(evg_ture_score_dict.values())[0]
        evg_pre_score = list(evg_pre_score_dict.values())[0]

        true_score_list.append(evg_ture_score)
        pred_score_list.append(evg_pre_score)




        det_metric.update(targets, outputs)
        seg_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    seg_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()
        seg_info = seg_metric.evaluate()
    else:
        coco_info = None
        seg_info = None

    true_scores = np.array(true_score_list)
    pred_scores = np.array(pred_score_list)
    #print("true_scores:", true_scores)
    #print("pred_scores:", pred_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    mse = mean_squared_error(pred_scores, true_scores)
    mae = mean_absolute_error(pred_scores, true_scores)
    rmse = np.sqrt(mse)
    result.rho = rho
    result.mae = mae
    result.rmse = rmse
    print(result)

    return coco_info, seg_info,result
