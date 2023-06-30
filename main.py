import prepare
import train

import optuna
import albumentations as A


if __name__ == "__main__":
    # BASE TRAIN, NO TUNNING===============================

    params = {
        "batch_size": 70,
        "dist_from_center": 15,
        "drop_out": 0.20879658201895385,
        "hidden_nodes": 256,
        "learning_rate": 0.0005881835636133336,
        "optimizer": "Adam",
    }
    train.hard_train_and_test(params)

    # TRAIN WITH TUNNING===============================
    study_name = "maximizing-accuracy"
    train.tune_parameters(1, study_name)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")

    train.hard_train_and_test(study.best_trial.params)

    # TRAIN WITH TUNNING AND ROTATE AUGMENT===============================
    transform = A.Compose(
        [A.Rotate(limit=35, p=0.2)],
    )
    study_name = "maximizing-accuracy-rotate"
    train.tune_parameters(1, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")

    train.hard_train_and_test(study.best_trial.params, transform)

    # TRAIN WITH TUNNING AND FLIP AUGMENT===============================
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
        ],
    )
    study_name = "maximizing-accuracy-flip"
    train.tune_parameters(1, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")

    train.hard_train_and_test(study.best_trial.params, transform)

    # TRAIN WITH TUNNING AND DISTORT AUGMENT===============================

    transform = A.Compose(
        [
            A.OpticalDistortion(p=0.2),
        ],
    )
    study_name = "maximizing-accuracy-distort"
    train.tune_parameters(1, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")

    train.hard_train_and_test(study.best_trial.params, transform)

    # TRAIN WITH TUNNING AND NOISE AUGMENT===============================

    transform = A.Compose(
        [
            A.GaussNoise(p=0.2),
        ],
    )
    study_name = "maximizing-accuracy-noise"
    train.tune_parameters(1, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")

    train.hard_train_and_test(study.best_trial.params, transform)

    # RECORDED BEST HYPERPARAMETERS BASED ON TUNNING RECORDS OF BASE,ROTATE,FLIP,DISTORT,NOISE ===============================

    # params = {
    #     "batch_size": 70,
    #     "dist_from_center": 15,
    #     "drop_out": 0.20879658201895385,
    #     "hidden_nodes": 256,
    #     "learning_rate": 0.0005881835636133336,
    #     "optimizer": "Adam",
    # }

    # params = {
    #     "batch_size": 50,
    #     "dist_from_center": 15,
    #     "drop_out": 0.17789086371197324,
    #     "hidden_nodes": 256,
    #     "learning_rate": 0.00017384951180295915,
    #     "optimizer": "Adam",
    # }

    # params = {
    #     "batch_size": 100,
    #     "dist_from_center": 20,
    #     "drop_out": 0.2836984186758677,
    #     "hidden_nodes": 256,
    #     "learning_rate": 0.0002686802819473838,
    #     "optimizer": "Adam",
    # }

    # params = {
    #     "batch_size": 100,
    #     "dist_from_center": 15,
    #     "drop_out": 0.28942132493859546,
    #     "hidden_nodes": 256,
    #     "learning_rate": 0.00045652817054764956,
    #     "optimizer": "Adam",
    # }

    # params = {
    #     "batch_size": 70,
    #     "dist_from_center": 20,
    #     "drop_out": 0.2716639026925446,
    #     "hidden_nodes": 256,
    #     "learning_rate": 0.00012934002493055115,
    #     "optimizer": "Adam",
    # }
