import csv
import os
import time

import torch
import torch.utils.data
import torch.distributed
import tqdm

import utils


def evaluate(model: torch.nn.Module,
             valloader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             num_classes: int,
             amp_enabled: bool,
             ddp_enabled: bool,
             device: torch.device) -> tuple[float, float, list, float]:
    model.eval()

    if ddp_enabled:
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 0

    evaluator = utils.metric.Evaluator(num_classes, device)
    inference_time = torch.zeros(1, device=device)
    val_loss = torch.zeros(1, device=device)
    for images, targets in tqdm.tqdm(valloader, desc='Eval', leave=False, disable=False if local_rank == 0 else True):
        images, targets = images.to(device), targets.to(device)

        with torch.cuda.amp.autocast(amp_enabled):
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                outputs = model(images)
            val_loss += criterion(outputs, targets)
            outputs = torch.argmax(outputs, dim=1)

            torch.cuda.synchronize()
            inference_time += time.time() - start_time

        # Update confusion matrix
        evaluator.update_matrix(targets, outputs)

    if ddp_enabled:
        val_loss_list = [val_loss]
        confusion_matrix_list = [evaluator.confusion_matrix]
        inference_time_list = [inference_time]
        torch.distributed.all_reduce_multigpu(val_loss_list, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce_multigpu(confusion_matrix_list, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce_multigpu(inference_time_list, op=torch.distributed.ReduceOp.SUM)

        val_loss = val_loss_list[0] / (len(valloader) * world_size)
        evaluator.confusion_matrix = confusion_matrix_list[0]
        mean_f1, f1 = evaluator.mean_f1_score(ignore_zero_class=True, percent=True)
        fps = len(valloader.dataset) / inference_time_list[0]
    else:
        val_loss /= len(valloader)
        mean_f1, f1 = evaluator.mean_f1_score(ignore_zero_class=True, percent=True)
        fps = len(valloader.dataset) / inference_time

    return val_loss.item(), mean_f1.item(), f1.tolist(), fps.item()


if __name__ == '__main__':
    # Create components builder
    builder = utils.builder.Builder()
    config = builder.config
    model_name = builder.model_name
    amp_enabled = config['train']['amp_enabled']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Dataset
    valset, valloader = builder.build_dataset('val')

    # 2. Model
    model = builder.build_model(valset.num_classes, pretrained=True).to(device)
    model.eval()
    print(f'Activated model: {model_name}')

    # 3. Loss function
    criterion = builder.build_criterion(device)

    # Evaluate model
    val_loss, mean_f1, f1, fps = evaluate(model, valloader, criterion, valset.num_classes, amp_enabled, False, device)

    # Save evaluation result as csv file
    os.makedirs('result', exist_ok=True)
    with open(os.path.join('result', f'{model_name}.csv'), mode='w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['Class Number', 'Class Name', 'F1'])

        assert len(valset.classes[1:]) == len(f1)
        for (name, id, _), f1_value in zip(valset.classes[1:], f1):
            writer.writerow([id, name, f1_value])
        writer.writerow(['mean F1', mean_f1, ' '])
        writer.writerow(['Validation loss', val_loss, ' '])
        writer.writerow(['FPS', fps, ' '])
    print('Saved evaluation result.')
