import optuna
import albumentations as A
import train
import prepare


def load_and_run_study(study_name, transform=None):
    train.tune_parameters(500, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    print(f"Best params: {study.best_trial.params}")

    return study


if __name__ == "__main__":
    # # Setting up
    # start_index = 80
    # stop_index = 200
    # test_files_list = [80, 120, 160]

    # params = {
    #     "batch_size": 128,
    #     "dist_from_center": 15,
    #     "drop_out": 0.28942132493859546,
    #     "hidden_nodes": 256,
    #     "learning_rate": 0.00045652817054764956,
    #     "optimizer": "Adam",
    # }

    # train.hard_train_and_test(params)

    # TRAIN WITH TUNNING AND DISTORT AUGMENT===============================
    print("TRAIN WITH TUNNING AND DISTORT AUGMENT")
    transform = A.Compose([A.OpticalDistortion(p=0.3)])
    study_name = "maximizing-accuracy-distort"
    study = load_and_run_study(study_name, transform)

    train.hard_train_and_test(study.best_trial.params, transform)
