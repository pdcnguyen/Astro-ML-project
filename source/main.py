import optuna
import albumentations as A
import train


def load_and_run_study(study_name, transform=None):
    train.tune_parameters(250, study_name, transform)
    study = optuna.load_study(study_name=study_name, storage="sqlite:///results.db")
    print(f"Best params: {study.best_trial.params}")

    return study


if __name__ == "__main__":
    # TRAIN WITH TUNNING AND DISTORT AUGMENT===============================
    print("TRAIN WITH TUNNING AND DISTORT AUGMENT")
    transform = A.Compose([A.OpticalDistortion(p=0.2)])
    study_name = "maximizing-accuracy-distort"
    study = load_and_run_study(study_name, transform)

    train.full_train_and_test(study.best_trial.params, transform)

    # Best params: {'batch_size': 32, 'drop_out': 0.29846878370550334, 'hidden_nodes': 256, 'learning_rate': 0.0003620066758014293, 'optimizer': 'Adam'}
