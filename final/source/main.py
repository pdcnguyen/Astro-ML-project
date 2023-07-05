import train as train

import optuna
import albumentations as A


def load_and_run_study(study_name, transform=None):
    train.tune_parameters(150, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    print(f'Best params: {study.best_trial.params}')

    return study


if __name__ == "__main__":
    # BASE TRAIN, NO TUNNING===============================
    # print("BASE TRAIN, NO TUNNING")
    params = {
        "batch_size": 50,
        "dist_from_center": 10,
        "drop_out": 0.3,
        "hidden_nodes": 512,
        "learning_rate": 0.0001,
        "optimizer": "RMSprop",
    }
    train.hard_train_and_test(params)

    # # TRAIN WITH TUNNING===============================
    print("TRAIN WITH TUNNING")
    study_name = "maximizing-accuracy"
    study = load_and_run_study(study_name)

    train.hard_train_and_test(study.best_trial.params)

    # TRAIN WITH TUNNING AND SHEAR AUGMENT===============================
    print("TRAIN WITH TUNNING AND SHEAR AUGMENT")
    transform = A.Compose([A.Affine(shear=(-45, 45))])
    study_name = "maximizing-accuracy-shear"
    study = load_and_run_study(study_name, transform)

    train.hard_train_and_test(study.best_trial.params, transform)

    # # TRAIN WITH TUNNING AND FLIP AUGMENT===============================
    print("TRAIN WITH TUNNING AND FLIP AUGMENT")
    transform = A.Compose([A.HorizontalFlip(p=0.2), A.VerticalFlip(p=0.2)])
    study_name = "maximizing-accuracy-flip"
    study = load_and_run_study(study_name, transform)

    train.hard_train_and_test(study.best_trial.params, transform)

    # # TRAIN WITH TUNNING AND DISTORT AUGMENT===============================
    print("TRAIN WITH TUNNING AND DISTORT AUGMENT")
    transform = A.Compose([A.OpticalDistortion(p=0.2)])
    study_name = "maximizing-accuracy-distort"
    study = load_and_run_study(study_name, transform)

    train.hard_train_and_test(study.best_trial.params, transform)

    # # TRAIN WITH TUNNING AND NOISE AUGMENT===============================
    print("TRAIN WITH TUNNING AND NOISE AUGMENT")
    transform = A.Compose([A.GaussNoise(p=0.2)])
    study_name = "maximizing-accuracy-noise"
    study = load_and_run_study(study_name, transform)

    train.hard_train_and_test(study.best_trial.params, transform)

# # RECORDED BEST HYPERPARAMETERS BASED ON TUNNING RECORDS OF BASE,ROTATE,FLIP,DISTORT,NOISE ===============================

# Using img_80: Test Loss: 0.457, Test Accuracy: 75.4%, Test f1: 0.61
# Using img_80: Test Loss: 0.213, Test Accuracy: 89.8%, Test f1: 0.80
# Using img_80: Test Loss: 0.246, Test Accuracy: 88.3%, Test f1: 0.79
# Using img_80: Test Loss: 0.222, Test Accuracy: 89.9%, Test f1: 0.80
# Using img_80: Test Loss: 0.198, Test Accuracy: 91.6%, Test f1: 0.82
# Using img_80: Test Loss: 0.698, Test Accuracy: 41.0%, Test f1: 0.25
