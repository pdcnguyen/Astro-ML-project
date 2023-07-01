import train

import optuna
import albumentations as A


if __name__ == "__main__":
    # BASE TRAIN, NO TUNNING===============================
    # print("BASE TRAIN, NO TUNNING")
    # params = {
    #     "batch_size": 50,
    #     "dist_from_center": 10,
    #     "drop_out": 0.3,
    #     "hidden_nodes": 512,
    #     "learning_rate": 0.0001,
    #     "optimizer": "RMSprop",
    # }
    # train.hard_train_and_test(params)

    # # TRAIN WITH TUNNING===============================
    # print("TRAIN WITH TUNNING")
    study_name = "maximizing-accuracy"
    # train.tune_parameters(150, study_name)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    print(study.best_trial)

    train.hard_train_and_test(study.best_trial.params)

    # # TRAIN WITH TUNNING AND SHEAR AUGMENT===============================
    # print("TRAIN WITH TUNNING AND SHEAR AUGMENT")
    # transform = A.Compose(
    #     [A.Affine(shear=(-45, 45))],
    # )
    # study_name = "maximizing-accuracy-shear"
    # train.tune_parameters(150, study_name, transform)
    # study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    # print(study.best_trial)

    # # train.hard_train_and_test(study.best_trial.params, transform)

    # # TRAIN WITH TUNNING AND FLIP AUGMENT===============================
    # print("TRAIN WITH TUNNING AND FLIP AUGMENT")
    # transform = A.Compose(
    #     [
    #         A.HorizontalFlip(p=0.2),
    #         A.VerticalFlip(p=0.2),
    #     ],
    # )
    # study_name = "maximizing-accuracy-flip"
    # train.tune_parameters(150, study_name, transform)
    # study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    # print(study.best_trial)

    # # train.hard_train_and_test(study.best_trial.params, transform)

    # # TRAIN WITH TUNNING AND DISTORT AUGMENT===============================
    # print("TRAIN WITH TUNNING AND DISTORT AUGMENT")
    # transform = A.Compose(
    #     [
    #         A.OpticalDistortion(p=0.2),
    #     ],
    # )
    # study_name = "maximizing-accuracy-distort"
    # train.tune_parameters(150, study_name, transform)
    # study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    # print(study.best_trial)

    # # train.hard_train_and_test(study.best_trial.params, transform)

    # # TRAIN WITH TUNNING AND NOISE AUGMENT===============================
    # print("TRAIN WITH TUNNING AND NOISE AUGMENT")
    # transform = A.Compose(
    #     [
    #         A.GaussNoise(p=0.2),
    #     ],
    # )
    # study_name = "maximizing-accuracy-noise"
    # train.tune_parameters(150, study_name, transform)
    # study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    # print(study.best_trial)

    # # train.hard_train_and_test(study.best_trial.params, transform)

    # # RECORDED BEST HYPERPARAMETERS BASED ON TUNNING RECORDS OF BASE,ROTATE,FLIP,DISTORT,NOISE ===============================
