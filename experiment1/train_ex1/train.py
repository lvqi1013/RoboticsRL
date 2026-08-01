import random
import numpy as np
import torch
import os
from torch import nn
from copy import deepcopy
import math
from torch import Tensor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

from train_config import RegressionLabelStats, CHEACKPOINT_OUTPUT_DIR
from models.lstm import LSTMRegressor
from models.mlp import MLPRegressor
from models.transformer import TransformerRegressor
from models.tabm import TabMModel
from geometry_metrics import subgoal_geometry_metrics


neural_factories = {
        "mlp": lambda d_in, d_out, _train_x: MLPRegressor(d_in, d_out),
        "lstm": lambda d_in, d_out, _train_x: LSTMRegressor(d_in, d_out),
        "transformer": lambda d_in, d_out, _train_x: TransformerRegressor(d_in, d_out),
        "tabm": lambda d_in, d_out, train_x: TabMModel(d_in, d_out, train_x)
    }

class Trainer:
    def __init__(self, model_name: str, 
                 epochs, lr,weight_decay,
                 parts: dict[str, np.ndarray],
                 label_stats: RegressionLabelStats,seed, device,
                 patience,
                 eval_every: int, 
                 batch_size: int,
                eval_batch_size: int,
                 map_size):
        
        self.set_seed(seed)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.map_size = map_size
        self.model_name = model_name
        self.label_stats = label_stats
        self.seed = seed
        self.patience = patience
        self.parts = parts
        self.eval_batch_size = eval_batch_size

        self.x_train = torch.as_tensor(parts["train_x"], device=device)
        self.y_train = torch.as_tensor(parts["train_y_norm"], device=device)
        self.x_val = torch.as_tensor(parts["val_x"], device=device)

        self.y_val_raw = parts["val_y"]
        self.x_test = torch.as_tensor(parts["test_x"], device=device)

        build_model = neural_factories[model_name]
        self.model: nn.Module = build_model(self.x_train.shape[1], self.y_train.shape[1], self.x_train).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=max(1, self.epochs))

        self.best_state = deepcopy(self.model.state_dict())
        self.best_score = math.inf
        self.patience_left = patience


    def run(self):
        for epoch in range(self.epochs):
            self.model.train()
            order = torch.randperm(len(self.x_train), device=self.device)
            for batch_idx in order.split(self.batch_size):
                pred = self.model(self.x_train[batch_idx]).float()
                if pred.ndim == 3:
                    pred = pred.flatten(0, 1)
                    target = self.y_train[batch_idx].repeat_interleave(pred.shape[0] // len(batch_idx), dim=0) 
                else:
                    target = self.y_train[batch_idx]

                loss = nn.functional.huber_loss(pred, target, delta=1.0)
                self.optimizer.zero_grad() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
                self.optimizer.step()
            self.scheduler.step()

            # ====================== 验证 & 早停逻辑 ======================
            if epoch % self.eval_every == 0 or epoch == self.epochs - 1:
                val_pred = self.predict_tensor(self.x_val)
                val_mse = mean_squared_error(self.y_val_raw, val_pred)
                val_rmse = math.sqrt(float(val_mse))

                if val_rmse < self.best_score:
                    self.best_score = val_rmse
                    self.best_state = deepcopy(self.model.state_dict())
                    self.patience_left = self.patience  # 指标提升，重置耐心
                else:
                    self.patience_left -= self.eval_every

                print(f"{self.model_name} seed={self.seed} epoch={epoch} val_rmse={val_rmse:.5f}, patience_left={self.patience_left}")

                if self.patience_left <= 0:
                    print(f"Early Stop! epoch={epoch}, best val_rmse={self.best_score:.5f}")
                    break
        
        self.model.load_state_dict(self.best_state)
        test_pred = self.predict_tensor(self.x_test)
        metrics = self.base_metrics(self.parts["test_y"], test_pred, self.parts["test_x_raw"], self.map_size)

        checkpoint = CHEACKPOINT_OUTPUT_DIR / f"{self.model_name}_seed{self.seed}.pt"
        torch.save(
        {
            "model": self.model_name,
            "model_state_dict": self.best_state,
            "label_stats": self.label_stats,
            "feature_count": int(self.parts["train_x"].shape[1]),
            "metrics": metrics,
        },
        checkpoint,
        )
        return metrics, checkpoint



    def base_metrics(self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: np.ndarray,
        map_size: int,
    )-> dict[str, float]:
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(math.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        }
        metrics.update(subgoal_geometry_metrics(features, y_true, y_pred, map_size))
        return metrics

        
            

    def predict_tensor(self,x_tensor: Tensor) -> np.ndarray:
        self.model.eval()
        preds = []

        with torch.no_grad():
            # TODO:这里的8192可以修改
            #  8192 不是 batch_size，是单块样本数量。显存充裕可放大，显存不足缩小。
            for batch in x_tensor.split(self.eval_batch_size):
                pred = self.model(batch).float()
                if pred.ndim == 3:
                    pred = pred.mean(dim=1)
                preds.append(pred.cpu().numpy())
        
        y_pred = np.concatenate(preds, axis=0)
        return y_pred * self.label_stats.std + self.label_stats.mean        



    def set_seed(self,seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def base_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: np.ndarray,
        map_size: int,
    )-> dict[str, float]:
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(math.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        }
        metrics.update(subgoal_geometry_metrics(features, y_true, y_pred, map_size))
        return metrics