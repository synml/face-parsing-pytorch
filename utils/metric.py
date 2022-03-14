import torch


class Evaluator:
    def __init__(self, num_classes: int, device: torch.device):
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        self.num_classes = num_classes
        # self.confusion_matrix.sum(dim=0): TP + FP
        # self.confusion_matrix.sum(dim=1): TP + FN

    def _generate_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor) -> torch.Tensor:
        label = self.num_classes * gt_batch + pred_batch
        bincount = torch.bincount(label.flatten(), minlength=self.num_classes ** 2)
        confusion_matrix = bincount.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def update_matrix(self, gt_batch: torch.Tensor, pred_batch: torch.Tensor):
        assert gt_batch.shape == pred_batch.shape
        self.confusion_matrix += self._generate_matrix(gt_batch, pred_batch)

    def pixel_accuracy(self) -> torch.Tensor:
        acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self) -> torch.Tensor:
        acc_cls = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(dim=0)
        acc_cls = torch.nanmean(acc_cls)
        return acc_cls

    def intersection_over_union(self, ignore_zero_class: bool) -> torch.Tensor:
        if ignore_zero_class:
            self.confusion_matrix = self.confusion_matrix[1:, 1:]

        iou = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=0) +
                                                   self.confusion_matrix.sum(dim=1) -
                                                   torch.diag(self.confusion_matrix))
        return iou

    def mean_intersection_over_union(self, ignore_zero_class: bool, percent=False) -> tuple[torch.Tensor, torch.Tensor]:
        iou = self.intersection_over_union(ignore_zero_class)

        if percent:
            iou *= 100

        miou = torch.nanmean(iou)
        return miou, iou

    # Same as dice coefficient
    def f1_score(self, ignore_zero_class: bool) -> torch.Tensor:
        if ignore_zero_class:
            self.confusion_matrix = self.confusion_matrix[1:, 1:]

        f1_score = torch.diag(self.confusion_matrix) * 2 / (self.confusion_matrix.sum(dim=0) +
                                                            self.confusion_matrix.sum(dim=1))
        return f1_score

    def mean_f1_score(self, ignore_zero_class: bool, percent=False) -> tuple[torch.Tensor, torch.Tensor]:
        f1_score = self.f1_score(ignore_zero_class)

        if percent:
            f1_score *= 100

        mean_f1_score = torch.nanmean(f1_score)
        return mean_f1_score, f1_score
