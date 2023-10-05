import numpy as np
from copy import deepcopy


class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf
        self.state_dict = None
        self.best_metric_score = 0

    def early_stop(self, model, loss, score):
        if loss < self.min_loss:
            self.min_loss = loss
            self.state_dict = deepcopy(model.state_dict())
            self.best_metric_score = score
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter += 0.5
            if self.counter >= self.patience:
                return True
        return False

    def load_best_model(self):
        return self.state_dict

    def get_best_metric_score(self):
        return self.best_metric_score
