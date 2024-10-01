import mlflow

from src.id_mapper import IDMapper


class MLflowLogCallback:
    def __init__(self):
        pass

    def process_payload(self, payload: dict):
        step = payload.get("step", None)
        dataset = payload.get("dataset", None)

        if dataset == "train":
            mlflow.log_metric("train_global_loss", payload["global_loss"], step=step)
            if "total_grad_norm" in payload:
                mlflow.log_metric(
                    "total_grad_norm", payload["total_grad_norm"], step=step
                )
            for key, value in payload.items():
                if key.startswith("grad_norm_"):
                    mlflow.log_metric(key, value, step=step)

        # Log validation loss at epoch level
        if "val_loss" in payload:
            mlflow.log_metric(
                "val_loss", payload["val_loss"], step=payload.get("epoch", 0)
            )

        # Log epoch-level metrics for training loss
        if "train_loss" in payload:
            mlflow.log_metric(
                "train_loss", payload["train_loss"], step=payload.get("epoch", 0)
            )

        if "total_train_time_seconds" in payload:
            mlflow.log_metrics(payload)

        if "learning_rate" in payload:
            mlflow.log_metric(
                "learning_rate", payload["learning_rate"], step=payload["epoch"]
            )


class MetricLogCallback:
    def __init__(self):
        self.payloads = []

    def process_payload(self, payload: dict):
        self.payloads.append(payload)


def log_gradients(model):
    total_norm = 0
    param_count = 0
    gradient_metrics = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            param_count += 1
            gradient_metrics[f"grad_norm_{name}"] = param_norm

    total_norm = total_norm**0.5
    gradient_metrics["total_grad_norm"] = total_norm
    return gradient_metrics


def map_indice(df, idm: IDMapper, user_col="user_id", item_col="parent_asin"):
    return df.assign(
        **{
            "user_indice": lambda df: df[user_col].apply(
                lambda user_id: idm.get_user_index(user_id)
            ),
            "item_indice": lambda df: df[item_col].apply(
                lambda item_id: idm.get_item_index(item_id)
            ),
        }
    )
