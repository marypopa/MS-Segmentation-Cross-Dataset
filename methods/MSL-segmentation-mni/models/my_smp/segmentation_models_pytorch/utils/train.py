import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
import re
import numpy as np
import os
from models.run_model import args

class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        slice_names = []

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y, slice_names_batch in iterator:
                x, y = x.to(self.device), y.to(self.device)
                slice_names = slice_names + slice_names_batch

                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.detach().cpu()
                # loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                # loss_logs = {"MY DICE": loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).detach().cpu()
                    # metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                l2_penalty = 0.5 * sum(torch.norm(param)**2 for param in self.model.parameters())
                aux_logs = {'l2': args.weight_decay * l2_penalty.detach().cpu().numpy().item()}
                logs.update(aux_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)


        # compute patient(i.e. scan)-level Dice scores
        patient_names = [re.split(r'_op_r_[0-9]*\.png', os.path.basename(slice_name))[0] for slice_name in slice_names]
        patients, I_patients = np.unique(patient_names, return_inverse=True)
        tp_list = metrics_meters["tp"].values
        fp_list = metrics_meters["fp"].values
        fn_list = metrics_meters["fn"].values
        tp_patient = torch.zeros(len(patients))
        fp_patient = torch.zeros(len(patients))
        fn_patient = torch.zeros(len(patients))
        for ip in range(len(patients)):
            Ipi = np.where(I_patients == ip)[0]
            tp_patient[ip] = torch.sum(tp_list[Ipi])
            fp_patient[ip] = torch.sum(fp_list[Ipi])
            fn_patient[ip] = torch.sum(fn_list[Ipi])
        dice_patient = 2*tp_patient / (2*tp_patient + fp_patient + fn_patient)
        dice_patient_average = torch.mean(dice_patient).detach().cpu().numpy().item()
        print('dice_patient_' + self.stage_name, dice_patient_average)
        logs.update({'dice_patient_' + self.stage_name: dice_patient_average})
        # print metrics computed over dataset
        iou_dataset = logs["tp"] / (logs["tp"] + logs["fp"] + logs["fn"])
        dice_dataset = 2*logs["tp"]/(2*logs["tp"] + logs["fp"] + logs["fn"])
        print('iou_dataset_' + self.stage_name, iou_dataset)
        print('dice_dataset_' + self.stage_name, dice_dataset)
        dataset_logs = {'iou_dataset_' + self.stage_name: iou_dataset, 'dice_dataset_' + self.stage_name: dice_dataset}
        logs.update(dataset_logs)
        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
