import numpy as np
from copy import deepcopy


class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.state_dict = None
        self.best_metric_score = 0

    def early_stop(self, validation_loss, model, score):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.state_dict = deepcopy(model.state_dict())
            self.best_metric_score = score
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif (
            validation_loss < (self.min_validation_loss + self.min_delta)
            and validation_loss > self.min_validation_loss
        ):
            self.counter += self.patience / 3.0
            if self.counter >= self.patience:
                return True
        return False

    def load_best_model(self):
        return self.state_dict

    def get_best_metric_score(self):
        return self.best_metric_score
