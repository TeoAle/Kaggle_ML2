from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


def plot_confusion_matrix(
        y, y_pred, style="tableau-colorblind10", plot_size=(8, 8)):

    cm = confusion_matrix(y, y_pred)

    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

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
