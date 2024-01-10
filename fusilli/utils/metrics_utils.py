"""
Calculates metrics of the models and houses list of the available metrics to use.
"""
import torch
import torchmetrics as tm


class MetricsCalculator:
    """
    Calculates metrics of the models and houses list of the available metrics to use.
    """

    def __init__(self, base_model_instance):
        """

        Parameters
        ----------
        base_model_instance : fusilli.fusionmodels.base_model.BaseModel
            Instance of the base model. Has information on the prediction task and multiclass dimensions if applicable.
        """
        self.model = base_model_instance
        self.prediction_task = base_model_instance.model.prediction_task

    def auroc(self, preds, labels, logits):
        """
        Area under the receiver operating characteristic curve.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            AUROC value.
        """

        if self.prediction_task == "binary":
            auroc_equation = tm.AUROC(task="binary")
        elif self.prediction_task == "multiclass":
            auroc_equation = tm.AUROC(num_classes=self.model.multiclass_dimensions, task="multiclass")
        else:
            raise ValueError("Invalid prediction task for AUROC.")

        return auroc_equation(logits, labels)

    def accuracy(self, preds, labels, logits):
        """
        Calculates accuracy.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            Accuracy value.

        """

        if self.prediction_task == "binary":
            # do binary accuracy
            accuracy_equation = tm.Accuracy(task="binary")

        elif self.prediction_task == "multiclass":
            # do multiclass accuracy
            accuracy_equation = tm.Accuracy(num_classes=self.model.multiclass_dimensions, task="multiclass", top_k=1)
        else:
            raise ValueError("Invalid prediction task for accuracy.")

        return accuracy_equation(preds, labels)

    def r2(self, preds, labels, logits):
        """
        Calculates R2 score.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            R2 score value.

        """

        if self.prediction_task != "regression":
            raise ValueError("Invalid prediction task for R2.")

        return tm.R2Score()(preds, labels)

    def mse(self, preds, labels, logits):
        """
        Calculates mean squared error.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            MSE value.

        """
        if self.prediction_task != "regression":
            raise ValueError("Invalid prediction task for mse.")

        return tm.MeanSquaredError()(preds, labels)

    def mae(self, preds, labels, logits):
        """
        Calculates mean absolute error.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            MAE value.

        """

        if self.prediction_task != "regression":
            raise ValueError("Invalid prediction task for mae.")

        return tm.MeanAbsoluteError()(preds, labels)

    def recall(self, preds, labels, logits):
        """
        Calculates recall. This is equivalent to sensitivity.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            Recall value.

        """

        if self.prediction_task == "binary":
            recall_equation = tm.Recall(task="binary")
        elif self.prediction_task == "multiclass":
            recall_equation = tm.Recall(num_classes=self.model.multiclass_dimensions, task="multiclass")
        else:
            raise ValueError("Invalid prediction task for recall.")

        return recall_equation(preds, labels)

    def specificity(self, preds, labels, logits):
        """
        Calculates specificity.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            Specificity value.

        """

        if self.prediction_task == "binary":
            specificity_equation = tm.Specificity(task="binary")
        elif self.prediction_task == "multiclass":
            specificity_equation = tm.Specificity(num_classes=self.model.multiclass_dimensions, task="multiclass")
        else:
            raise ValueError("Invalid prediction task for specificity.")

        return specificity_equation(preds, labels)

    def precision(self, preds, labels, logits):
        """
        Calculates precision.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            Precision value.

        """

        if self.prediction_task == "binary":
            precision_equation = tm.Precision(task="binary")
        elif self.prediction_task == "multiclass":
            precision_equation = tm.Precision(num_classes=self.model.multiclass_dimensions, task="multiclass")
        else:
            raise ValueError("Invalid prediction task for precision.")

        return precision_equation(preds, labels)

    def f1(self, preds, labels, logits):
        """
        Calculates F1 score. This is equivalent to the Dice coefficient.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            F1 score value.

        """

        if self.prediction_task == "binary":
            f1_equation = tm.F1Score(task="binary")
        elif self.prediction_task == "multiclass":
            f1_equation = tm.F1Score(num_classes=self.model.multiclass_dimensions, task="multiclass")
        else:
            raise ValueError("Invalid prediction task for F1.")

        return f1_equation(preds, labels)

    def auprc(self, preds, labels, logits):
        """
        Calculates area under the precision-recall curve.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            AUPRC value.
        """

        if self.prediction_task == "binary":
            auprc_equation = tm.AveragePrecision(task="binary")
        elif self.prediction_task == "multiclass":
            auprc_equation = tm.AveragePrecision(num_classes=self.model.multiclass_dimensions, task="multiclass")
        else:
            raise ValueError("Invalid prediction task for AUPRC.")

        return auprc_equation(logits, labels)

    def balanced_accuracy(self, preds, labels, logits):
        """
        Calculates balanced accuracy.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted values from the model.
        labels : torch.Tensor
            True labels.
        logits : torch.Tensor
            Probability values from the model.

        Returns
        -------
        float
            Balanced accuracy value.
        """

        if self.prediction_task == "binary":
            balanced_accuracy_equation = tm.Accuracy(task='multiclass', num_classes=2, average='macro')
        elif self.prediction_task == "multiclass":
            balanced_accuracy_equation = tm.Accuracy(task='multiclass', num_classes=self.model.multiclass_dimensions,
                                                     average='macro')
        else:
            raise ValueError("Invalid prediction task for balanced accuracy.")

        return balanced_accuracy_equation(preds, labels)
