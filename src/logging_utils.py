from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import mlflow


def plot_confusion_matrix(
        y, y_pred, plot_size=(8, 8)):

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.close(fig)
    return fig


def champion_callback(study, frozen_trial):

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value) * 100

            print(
                f"Trial {frozen_trial.number} value: {frozen_trial.value} with"
                f" {improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Trial {frozen_trial.number} value: {frozen_trial.value}")


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
