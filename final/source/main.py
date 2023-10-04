import optuna
import albumentations as A
import train
import prepare


def load_and_run_study(study_name, transform=None):
    train.tune_parameters(200, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    print(f"Best params: {study.best_trial.params}")

    return study


if __name__ == "__main__":
    # Setting up
    start_index = 80
    stop_index = 160
    test_files_list = [80, 120, 160]

    prepare.prepate_data(start_index, stop_index, test_files_list)

    exit(0)

    # # TRAIN WITH TUNNING===============================
    print("TRAIN WITH TUNNING")
    study_name = "maximizing-accuracy"
    study = load_and_run_study(study_name)

    train.hard_train_and_test(study.best_trial.params)

    # TRAIN WITH TUNNING AND SHEAR AUGMENT===============================
    print("TRAIN WITH TUNNING AND SHEAR AUGMENT")
    transform = A.Compose([A.Affine(shear=(-45, 45), p=0.2)])
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
