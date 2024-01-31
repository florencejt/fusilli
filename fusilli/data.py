"""
Data loading classes for multimodal and unimodal data.
This file contains functions and classes for loading the data for the different modalities, and
in training the subspace methods on the data (if the subspace methods need pre-training).
Train/test splits and k-fold cross validation are also implemented here.
"""

# imports
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data.lightning import LightningNodeData
from fusilli.utils import model_modifier


def downsample_img_batch(imgs, output_size):
    """
    Downsamples a batch of images to a specified size.

    Parameters
    ----------
    imgs : array-like
        Batch of images. Shape (batch_size, channels, height, width) or (batch_size, channels,
        height, width, depth) for 3D images.

    output_size : tuple
        Size to downsample the images to (height, width) or (height, width, depth) for 3D images.
        Do not put the batch_size dimension in the tuple.
        If None, no downsampling is performed

    Returns
    -------
    downsampled_img : array-like
        Downsampled image.
    """

    if output_size is None:  # if no downsampling
        return imgs

    # if number of output_size dims is equal to number of image dims - 3
    # (i.e. if output_size is (64) and image is (16, 3, 128, 128))
    # or output_size is (64, 64) and image is (16, 3, 128, 128, 128))

    if len(output_size) == imgs.dim() - 3:
        raise ValueError(
            f"output_size must have {imgs.dim() - 3} dimensions, not {len(output_size)}.\
                Make sure to exclude the channel dimension so output_size looks like\
                    (height, width) for 2D or (height, width, depth) for 3D."
        )

    # if output_size has a negative value
    if any([i < 0 for i in output_size]):
        raise ValueError(
            f"output_size must not have negative values, but got {output_size}."
        )

    # if output_size has more than 3 dimensions
    if len(output_size) > 3:
        raise ValueError(
            f"output_size must have 2 or 3 dimensions, not {len(output_size)}."
        )

    # if output_size has more than 2 dimensions and image is 2D
    if len(output_size) > 2 and imgs.dim() == 4:
        raise ValueError(
            f"output_size must have 2 dimensions, not {len(output_size)} because img_dims indicates a 2D image."
        )

    # if output_size has more than 4 dimensions and image is 3D
    if len(output_size) > 3 and imgs.dim() == 5:
        raise ValueError(
            f"output_size must have 3 dimensions, not {len(output_size)} because img_dims indicates a 3D image."
        )

    # if output_size is larger than image dimensions
    if any([i > j for i, j in zip(output_size, imgs.shape[2:])]):
        raise ValueError(
            f"output_size must be smaller than image dimensions, but got {output_size} and "
            f"image dimensions {imgs.shape[2:]}"
        )

    downsampled_img = F.interpolate(imgs, size=output_size, mode="nearest")

    return downsampled_img


class CustomDataset(Dataset):
    """
    Custom dataset class for multimodal data.

    Attributes
    ----------
    multimodal_flag : bool
        Flag for multimodal data. True if multimodal, False if unimodal.
    dataset1 : tensor
        Tensor of predictive features for modality 1.
    dataset2 : tensor
        Tensor of predictive features for modality 2.
    dataset : tensor
        Tensor of predictive features for uni-modal data.
    labels : tensor
        Tensor of labels.

    """

    def __init__(self, pred_features, labels):
        """
        Parameters
        ----------

        pred_features : list or tensor
            List of tensors or tensor of predictive features
            (i.e. tabular or image data without labels).
        labels : dataframe
            Dataframe of labels (column name must be "prediction_label").

        Raises
        ------
        ValueError
            If pred_features is not a list or tensor.

        """

        # if pred_features is a list: it's multimodal data
        # only 2 modalities are supported currently
        if isinstance(pred_features, list):
            self.multimodal_flag = True
            self.dataset1 = pred_features[0].float()
            self.dataset2 = pred_features[1].float()

        # if pred_features is a tensor: it's unimodal data
        elif isinstance(pred_features, torch.Tensor):
            self.multimodal_flag = False
            self.dataset = pred_features.float()

        else:
            raise ValueError(
                f"pred_features must be a list or a tensor, not {type(pred_features)}"
            )

        # convert labels to tensor and correct dtype
        label_type = labels[["prediction_label"]].values.dtype
        self.labels = torch.tensor(labels[["prediction_label"]].to_numpy().reshape(-1))
        if label_type == "int64":
            self.labels = self.labels.long()
        else:
            self.labels = self.labels.float()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        if self.multimodal_flag:
            return len(self.dataset1)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        Parameters
        ----------
        idx : int
            Index of the item to return.
        """
        if self.multimodal_flag:
            return self.dataset1[idx], self.dataset2[idx], self.labels[idx]
        else:
            return self.dataset[idx], self.labels[idx]


class LoadDatasets:
    """
    Class for loading the different datasets for the different modalities.

    Attributes
    ----------
    tabular1_source : str
        Source csv file for tabular1 data.
    tabular2_source : str
        Source csv file for tabular2 data.
    img_source : str
        Source torch file for image data.
    image_downsample_size : tuple
        Size to downsample the images to (height, width, depth) or (height, width) for 2D
        images.
        None if not downsampling. (default None)
    """

    def __init__(self, sources, img_downsample_dims=None):
        """
        Parameters
        ----------
        sources : list
            List of source csv files.
            [tabular1_source, tabular2_source, img_source]
        img_downsample_dims : tuple
            Size to downsample the images to (height, width, depth) or (height, width) for 2D
            images.
            None if not downsampling. (default None)

        Raises
        ------
        ValueError
            If sources is not a list.
        ValueError
            If the CSVs do not have the right columns or if the index column is not named
            "ID".

        """
        self.tabular1_source, self.tabular2_source, self.img_source = sources
        self.image_downsample_size = (
            img_downsample_dims  # can choose own image size here
        )

        # read in the csv files and raise errors if they don't have the right columns
        # or if the index column is not named "ID"
        tab1_df = pd.read_csv(self.tabular1_source)
        if "ID" not in tab1_df.columns:
            raise ValueError("The CSV must have an index column named 'ID'.")
        if "prediction_label" not in tab1_df.columns:
            raise ValueError("The CSV must have a label column named 'prediction_label'.")

        # if tabular2_source exists, check it has the right columns
        if self.tabular2_source != "":
            tab2_df = pd.read_csv(self.tabular2_source)
            if "ID" not in tab2_df.columns:
                raise ValueError("The CSV must have an index column named 'ID'.")
            if "prediction_label" not in tab2_df.columns:
                raise ValueError("The CSV must have a label column named 'prediction_label'.")

    def load_tabular1(self):
        """
        Loads the tabular1-only dataset

        Returns
        ------
        dataset (tensor): tensor of predictive features
        data_dims (list): list of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image

        """
        tab_df = pd.read_csv(self.tabular1_source)

        tab_df.set_index("ID", inplace=True)

        pred_features = torch.Tensor(tab_df.drop(columns=["prediction_label"]).values)
        prediction_label = tab_df[["prediction_label"]]

        dataset = CustomDataset(pred_features, prediction_label)

        mod1_dim = pred_features.shape[1]

        return dataset, [mod1_dim, None, None]

    def load_tabular2(self):
        """
        Loads the tabular2-only dataset

        Returns
        ------
        dataset (tensor): tensor of predictive features
        data_dims (list): list of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image
        """

        tab_df = pd.read_csv(self.tabular2_source)

        tab_df.set_index("ID", inplace=True)

        pred_features = torch.Tensor(tab_df.drop(columns=["prediction_label"]).values)
        prediction_label = tab_df[["prediction_label"]]

        dataset = CustomDataset(pred_features, prediction_label)
        mod2_dim = pred_features.shape[1]

        return dataset, [None, mod2_dim, None]

    def load_img(self):
        """
        Loads the image-only dataset

        Returns
        ------
        dataset (tensor): tensor of predictive features
        data_dims (list): list of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image
        """

        all_scans = torch.load(self.img_source)
        all_scans_ds = downsample_img_batch(all_scans, self.image_downsample_size)

        # get the labels from the tabular1 dataset
        label_df = pd.read_csv(self.tabular1_source)

        label_df.set_index("ID", inplace=True)

        prediction_label = label_df[["prediction_label"]]

        dataset = CustomDataset(all_scans_ds, prediction_label)

        img_dim = list(all_scans_ds.shape[2:])  # not including batch size or channels

        return dataset, [None, None, img_dim]

    def load_tabular_tabular(self):
        """
        Loads the tabular1 and tabular2 multimodal dataset

        Returns
        ------
        dataset (tensor): tensor of predictive features
        data_dims (list): list of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image
        """

        tab1_df = pd.read_csv(self.tabular1_source)
        tab2_df = pd.read_csv(self.tabular2_source)

        tab1_df.set_index("ID", inplace=True)
        tab2_df.set_index("ID", inplace=True)

        tab1_pred_features = torch.Tensor(tab1_df.drop(columns=["prediction_label"]).values)
        tab2_pred_features = torch.Tensor(tab2_df.drop(columns=["prediction_label"]).values)

        prediction_label = tab1_df[["prediction_label"]]
        dataset = CustomDataset([tab1_pred_features, tab2_pred_features], prediction_label)

        mod1_dim = tab1_pred_features.shape[1]
        mod2_dim = tab2_pred_features.shape[1]

        return dataset, [mod1_dim, mod2_dim, None]

    def load_tab_and_img(self):
        """
        Loads the tabular1 and image multimodal dataset.

        Returns
        ------
        dataset (tensor): tensor of predictive features
        data_dims (list): list of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image
        """

        tab1_df = pd.read_csv(self.tabular1_source)

        tab1_df.set_index("ID", inplace=True)

        tab1_features = torch.Tensor(tab1_df.drop(columns=["prediction_label"]).values)
        label_df = tab1_df[["prediction_label"]]

        imgs = torch.load(self.img_source)
        imgs = downsample_img_batch(imgs, self.image_downsample_size)

        dataset = CustomDataset([tab1_features, imgs], label_df)
        mod1_dim = tab1_features.shape[1]
        img_dim = list(imgs.shape[2:])  # not including batch size or channels

        return dataset, [mod1_dim, None, img_dim]


class TrainTestDataModule(pl.LightningDataModule):
    """
    Custom pytorch lightning datamodule class for the different modalities.

    Attributes
    ----------
    sources : list
        List of source csv files. [Tabular1, Tabular2, Image]
    modality_methods : dict
        Dictionary of methods for loading the different modalities.
    fusion_model : class
        fusion model class. e.g. TabularCrossmodalAttention.
    output_paths : dict
        Dictionary of output paths for saving the checkpoints, figures, and the losses.
    batch_size : int
        Batch size (default 8).
    test_size : float
        Fraction of data to use for testing (default 0.2).
    prediction_task : str
        Prediction type (binary, multiclass, or regression).
    multiclass_dimensions : int
        Number of classes for multiclass prediction (default None).
    subspace_method : class
        Subspace method class (default None) (only for subspace methods).
    layer_mods : dict
        Dictionary of layer modifications to make to the subspace method.
        (default None)
    max_epochs : int
        Maximum number of epochs to train subspace methods for. (default 1000)
    dataset : tensor
        Tensor of predictive features. Created in prepare_data().
    data_dims : list
        List of data dimensions [mod1_dim, mod2_dim, img_dim].
        Created in prepare_data().
    train_dataset : tensor
        Tensor of predictive features for training. Created in setup().
    test_dataset : tensor
        Tensor of predictive features for testing. Created in setup().
    subspace_method_train : class
        Subspace method class trained (only for subspace methods).
    own_early_stopping_callback : pytorch_lightning.callbacks.EarlyStopping
        Early stopping callback class.
    num_workers : int
        Number of workers for the dataloader (default 0).
    test_indices : list
        List of indices to use for testing (default None). If None, the test indices are
        randomly selected using the test_size parameter.
    kwargs : dict
        Dictionary of extra arguments for the subspace method class.
    """

    def __init__(
            self,
            fusion_model,
            sources,
            output_paths,
            prediction_task,
            batch_size,
            test_size,
            multiclass_dimensions,
            subspace_method=None,
            image_downsample_size=None,
            layer_mods=None,
            max_epochs=1000,
            extra_log_string_dict=None,
            own_early_stopping_callback=None,
            num_workers=0,
            test_indices=None,
            kwargs=None,
    ):
        """
        Parameters
        ----------

        fusion_model : class
            Fusion model class. e.g. "TabularCrossmodalAttention".
        sources : list
            List of source csv files.
        output_paths : dict
            Dictionary of output paths for saving the checkpoints, figures, and the losses.
        prediction_task : str
            Prediction task (binary, multiclass, regression).
        batch_size : int
            Batch size (default 8).
        test_size : float
            Fraction of data to use for testing (default 0.2).
        multiclass_dimensions : int
            Number of classes for multiclass prediction (default None).
        subspace_method : class
            Subspace method class (default None) (only for subspace methods).
        image_downsample_size : tuple
            Size to downsample the images to (height, width, depth) or (height, width) for 2D
            images.
            None if not downsampling. (default None)
        layer_mods : dict
            Dictionary of layer modifications to make to the subspace method.
            (default None)
        max_epochs : int
            Maximum number of epochs to train subspace methods for. (default 1000)
        extra_log_string_dict : dict
            Dictionary of extra strings to add to the log.
        own_early_stopping_callback : pytorch_lightning.callbacks.EarlyStopping
            Early stopping callback class (default None).
        num_workers : int
            Number of workers for the dataloader (default 0).
        test_indices : list
            List of indices to use for testing (default None). If None, the test indices are
            randomly selected using the test_size parameter.
        kwargs : dict
            Dictionary of extra arguments for the subspace method class.
        """
        super().__init__()

        self.sources = sources
        self.output_paths = output_paths
        self.extra_log_string_dict = extra_log_string_dict
        self.modality_methods = {
            "tabular1": LoadDatasets(self.sources, image_downsample_size).load_tabular1,
            "tabular2": LoadDatasets(self.sources, image_downsample_size).load_tabular2,
            "img": LoadDatasets(self.sources, image_downsample_size).load_img,
            "tabular_tabular": LoadDatasets(
                self.sources, image_downsample_size
            ).load_tabular_tabular,
            "tabular_image": LoadDatasets(
                self.sources, image_downsample_size
            ).load_tab_and_img,
        }
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.batch_size = batch_size
        self.test_size = test_size
        self.prediction_task = prediction_task
        if self.prediction_task == "multiclass":
            self.multiclass_dimensions = multiclass_dimensions
        else:
            self.multiclass_dimensions = None
        self.subspace_method = subspace_method
        self.layer_mods = layer_mods
        self.max_epochs = max_epochs
        self.own_early_stopping_callback = own_early_stopping_callback
        self.num_workers = num_workers
        self.test_indices = test_indices
        self.kwargs = kwargs

    def prepare_data(self):
        """
        Loads the data with LoadDatasets class

        Returns
        ------
        dataset : tensor
            Tensor of predictive features.
        data_dims : list
            List of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image
        """
        self.dataset, self.data_dims = self.modality_methods[self.modality_type]()

    def setup(
            self,
            checkpoint_path=None,
    ):
        """
        Splits the data into train and test sets, and runs the subspace method if specified.
        If checkpoint_path is specified, the subspace method is loaded from the checkpoint and not trained.

        Attributes
        ----------
        checkpoint_path : str
            Path to the checkpoint file for the subspace method (default None).

        Returns
        ------
        train_dataloader : dataloader
            Dataloader for training.
        val_dataloader : dataloader
            Dataloader for validation.
        """

        # split the dataset into train and test sets
        if self.test_indices is None:
            [self.train_dataset, self.test_dataset] = torch.utils.data.random_split(
                self.dataset, [1 - self.test_size, self.test_size]
            )
        else:
            self.test_dataset = torch.utils.data.Subset(
                self.dataset, self.test_indices
            )

            self.train_dataset = torch.utils.data.Subset(
                self.dataset, list(set(range(len(self.dataset))) - set(self.test_indices))
            )

        if self.subspace_method is not None:  # if subspace method is specified
            if (
                    checkpoint_path is None
            ):  # if no checkpoint path specified, train the subspace method
                self.subspace_method_train = self.subspace_method(
                    datamodule=self,
                    max_epochs=self.max_epochs,
                    k=None,
                    train_subspace=True
                )

                # modify the subspace method architecture if specified
                if self.layer_mods is not None:
                    self.subspace_method_train = model_modifier.modify_model_architecture(
                        self.subspace_method_train,
                        self.layer_mods,
                    )

                # train the subspace method and convert train dataset to the latent space
                train_latents, train_labels = self.subspace_method_train.train(
                    self.train_dataset, self.test_dataset
                )

                # convert the test dataset to the latent space
                (
                    test_latents,
                    test_labels,
                    data_dims,
                ) = self.subspace_method_train.convert_to_latent(self.test_dataset)

                # create the new train and test datasets from the latent space with updated dimensions
                self.train_dataset = CustomDataset(train_latents, train_labels)
                self.test_dataset = CustomDataset(test_latents, test_labels)
                self.data_dims = data_dims

            else:
                # we have already trained the subspace method, so load it from the checkpoint

                self.subspace_method_train = self.subspace_method(
                    self,
                    max_epochs=self.max_epochs,
                    k=None,
                    train_subspace=False
                )  # will return a init subspace method with the subspace models as instance attributes

                # modify the subspace method architecture if specified
                if self.layer_mods is not None:
                    self.subspace_method_train = model_modifier.modify_model_architecture(
                        self.subspace_method_train,
                        self.layer_mods,
                    )

                # load checkpoint state dict
                self.subspace_method_train.load_ckpt(checkpoint_path)

                # converting the train and test datasets to the latent space
                (
                    train_latents,
                    train_labels,
                    data_dims,
                ) = self.subspace_method_train.convert_to_latent(self.train_dataset)

                (
                    test_latents,
                    test_labels,
                    data_dims,
                ) = self.subspace_method_train.convert_to_latent(self.test_dataset)

                # create the new train and test datasets from the latent space with updated dimensions
                self.train_dataset = CustomDataset(train_latents, train_labels)
                self.test_dataset = CustomDataset(test_latents, test_labels)
                self.data_dims = data_dims

    def train_dataloader(self):
        """
        Returns the dataloader for training.

        Returns
        -------
        dataloader : dataloader
            Dataloader for training.
        """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """
        Returns the dataloader for validation.

        Returns
        -------
        dataloader : dataloader
            Dataloader for validation.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )


class KFoldDataModule(pl.LightningDataModule):
    """
    Custom pytorch lightning datamodule class for the different modalities with k-fold cross
    validation

    Attributes
    ----------
    num_folds : int
        Total number of folds.
    sources : list
        List of source csv files. [Tabular1, Tabular2, Image]
    output_paths : dict
        Dictionary of output paths for saving the checkpoints, figures, and the losses.
    image_downsample_size : tuple
        Size to downsample the images to (height, width, depth) or (height, width) for 2D
        images.
        None if not downsampling. (default None)
    modality_methods : dict
        Dictionary of methods for loading the different modalities.
    fusion_model : class
        Fusion model class. e.g. "TabularCrossmodalAttention".
    batch_size : int
        Batch size (default 8).
    prediction_task : str
        Prediction type (binary, multiclass, regression).
    multiclass_dimensions : int
        Number of classes for multiclass prediction (default None).
    subspace_method : class
        Subspace method class (default None) (only for subspace methods).
    layer_mods : dict
        Dictionary of layer modifications to make to the subspace method.
        (default None)
    max_epochs : int
        Maximum number of epochs to train subspace methods for. (default 1000)
    dataset : tensor
        Tensor of predictive features. Created in prepare_data().
    data_dims : list
        List of data dimensions [mod1_dim, mod2_dim, img_dim]. Created in prepare_data().
    train_dataset : tensor
        Tensor of predictive features for training. Created in setup().
    test_dataset : tensor
        Tensor of predictive features for testing. Created in setup().
    own_early_stopping_callback : pytorch_lightning.callbacks.EarlyStopping
        Early stopping callback class.
    num_workers : int
        Number of workers for the dataloader (default 0).
    own_kfold_indices : list
        List of indices to use for k-fold cross validation (default None). If None, the k-fold
        indices are randomly selected. Structure is a list of tuples of (train_indices,
        test_indices). Must be the same length as num_folds.
    kwargs : dict
        Dictionary of extra arguments for the subspace method class.
    """

    def __init__(
            self,
            fusion_model,
            sources,
            output_paths,
            prediction_task,
            batch_size,
            num_folds,
            multiclass_dimensions,
            subspace_method=None,
            image_downsample_size=None,
            layer_mods=None,
            max_epochs=1000,
            extra_log_string_dict=None,
            own_early_stopping_callback=None,
            num_workers=0,
            own_kfold_indices=None,
            kwargs=None,
    ):
        """
        Parameters
        ----------

        fusion_model : class
            Fusion model class. e.g. "TabularCrossmodalAttention".
        sources : list
            List of source data files: csv or torch files.
        output_paths : dict
            Dictionary of output paths for saving the checkpoints, figures, and the losses.
        prediction_task : str
            Prediction task (binary, multiclass, regression).
        batch_size : int
            Batch size.
        num_folds : int
            Total number of folds.
        test_size : float
            Fraction of data to use for testing (default 0.2).
            Not needed for this class for k-fold cross validation but it's here to be consistent with TrainTestDataModule.
        multiclass_dimensions : int
            Number of classes for multiclass prediction (default None).
        subspace_method : class
            Subspace method class (default None) (only for subspace methods).
        image_downsample_size : tuple
            Size to downsample the images to (height, width, depth) or (height, width) for 2D
            images. None if not downsampling. (default None)
        layer_mods : dict
            Dictionary of layer modifications to make to the subspace method.
            (default None)
        max_epochs : int
            Maximum number of epochs to train subspace methods for. (default 1000)
        extra_log_string_dict : dict
            Dictionary of extra strings to add to the log.
        own_early_stopping_callback : pytorch_lightning.callbacks.EarlyStopping
            Early stopping callback class (default None).
        num_workers : int
            Number of workers for the dataloader (default 0).
        own_kfold_indices : list
            List of indices to use for k-fold cross validation (default None). If None, the k-fold
            indices are randomly selected. Structure is a list of tuples of (train_indices,
            test_indices). Must be the same length as num_folds.
        kwargs : dict
            Dictionary of extra arguments for the subspace method class.
        """
        super().__init__()

        self.num_folds = num_folds  # total number of folds
        self.sources = sources
        self.output_paths = output_paths
        self.image_downsample_size = image_downsample_size
        self.extra_log_string_dict = extra_log_string_dict
        self.modality_methods = {
            "tabular1": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular1,
            "tabular2": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular2,
            "img": LoadDatasets(self.sources, self.image_downsample_size).load_img,
            "tabular_tabular": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular_tabular,
            "tabular_image": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tab_and_img,
        }
        self.prediction_task = prediction_task
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.batch_size = batch_size
        self.subspace_method = (
            subspace_method  # subspace method class (only for subspace methods)
        )
        if self.prediction_task == "multiclass":
            self.multiclass_dimensions = multiclass_dimensions
        else:
            self.multiclass_dimensions = None
        self.layer_mods = layer_mods
        self.max_epochs = max_epochs
        self.own_early_stopping_callback = own_early_stopping_callback
        self.num_workers = num_workers
        self.own_kfold_indices = own_kfold_indices
        self.kwargs = kwargs

    def prepare_data(self):
        """
        Loads the data with LoadDatasets class

        Returns
        ------
        dataset : tensor
            Tensor of predictive features.
        data_dims : list
            List of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image
        """
        self.dataset, self.data_dims = self.modality_methods[self.modality_type]()

    def kfold_split(self):
        """
        Splits the dataset into k folds

        Returns
        ------
        folds : list
            List of tuples of (train_dataset, test_dataset)
        """
        # get the indices of the dataset
        indices = list(range(len(self.dataset)))

        # split the dataset into k folds
        if self.own_kfold_indices is None:
            kf = KFold(n_splits=self.num_folds, shuffle=True)
            split_kf = kf.split(indices)
        else:
            split_kf = self.own_kfold_indices

        folds = []
        for train_indices, val_indices in split_kf:
            # split the dataset into train and test sets for each fold
            train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            test_dataset = torch.utils.data.Subset(self.dataset, val_indices)

            # append the train and test datasets to the folds list
            folds.append((train_dataset, test_dataset))

        return folds  # list of tuples of (train_dataset, test_dataset)

    def setup(
            self,
            checkpoint_path=None,
    ):
        """
        Splits the data into train and test sets, and runs the subspace method if specified

        Attributes
        ----------
        checkpoint_path : str
            Path to the checkpoint file for the subspace method (default None).

        Returns
        ------
        train_dataloader : dataloader
            Dataloader for training.
        val_dataloader : dataloader
            Dataloader for validation.
        """
        self.folds = self.kfold_split()  # get the k folds from kfold_split() function

        # if subspace method is specified, run the subspace method on each fold
        if self.subspace_method is not None:
            # if no checkpoint path specified, train the subspace method
            if checkpoint_path is None:
                new_folds = []

                for k, fold in enumerate(self.folds):
                    # get the train and test datasets for each fold
                    train_dataset, test_dataset = fold

                    #  initialise the subspace method
                    subspace_method = self.subspace_method(
                        self,
                        k=k,
                        max_epochs=self.max_epochs,
                        train_subspace=True,
                    )

                    # modify the subspace method architecture if specified
                    if self.layer_mods is not None:
                        # if subspace method in layer_mods
                        subspace_method = model_modifier.modify_model_architecture(
                            subspace_method,
                            self.layer_mods,
                        )

                    # train the subspace method and convert train dataset to the latent space
                    train_latents, train_labels = subspace_method.train(
                        train_dataset, test_dataset
                    )

                    # convert the test dataset to the latent space
                    (
                        test_latents,
                        test_labels,
                        data_dims,
                    ) = subspace_method.convert_to_latent(test_dataset)

                    # make a new CustomDataset with the latent features
                    train_dataset = CustomDataset(train_latents, train_labels)
                    test_dataset = CustomDataset(test_latents, test_labels)

                    new_folds.append(
                        (train_dataset, test_dataset)
                    )  # append to new_folds

                self.folds = (
                    new_folds  # update the folds with the new train and test datasets
                )
                self.data_dims = data_dims  # update the data dimensions

            else:  # we have already trained the subspace method, so load it from the checkpoint
                new_folds = []

                for k, fold in enumerate(self.folds):
                    # get the train and test datasets for each fold
                    train_dataset, test_dataset = fold

                    #  initialise the subspace method
                    subspace_method = self.subspace_method(
                        self,
                        k=k,
                        max_epochs=self.max_epochs,
                        train_subspace=False,
                        # checkpoint_path=checkpoint_path,
                    )

                    # modify the subspace method architecture if specified
                    if self.layer_mods is not None:
                        # if subspace method in layer_mods
                        subspace_method = model_modifier.modify_model_architecture(
                            subspace_method,
                            self.layer_mods,
                        )
                    subspace_method.load_ckpt(checkpoint_path)

                    (
                        train_latents,
                        train_labels,
                        data_dims,
                    ) = subspace_method.convert_to_latent(train_dataset)

                    # convert the test dataset to the latent space
                    (
                        test_latents,
                        test_labels,
                        data_dims,
                    ) = subspace_method.convert_to_latent(test_dataset)

                    # make a new CustomDataset with the latent features
                    train_dataset = CustomDataset(train_latents, train_labels)
                    test_dataset = CustomDataset(test_latents, test_labels)

                    new_folds.append((train_dataset, test_dataset))

                self.folds = (
                    new_folds  # update the folds with the new train and test datasets
                )
                self.data_dims = data_dims  # update the data dimensions

    def train_dataloader(self, fold_idx):
        """
        Returns the dataloader for training.

        Parameters
        ----------
        fold_idx : int
            Index of the fold to use.

        Returns
        -------
        dataloader : dataloader
            Dataloader for training.
        """
        self.train_dataset, self.test_dataset = self.folds[fold_idx]

        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self, fold_idx):
        """
        Returns the dataloader for validation.

        Parameters
        ----------
        fold_idx : int
            Index of the fold to use.

        Returns
        -------
        dataloader : dataloader
            Dataloader for validation.
        """
        self.train_dataset, self.test_dataset = self.folds[fold_idx]

        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )


class TrainTestGraphDataModule:
    """
    Custom pytorch lightning datamodule class for the different modalities with graph data
    structure.

    Attributes
    ----------
    sources : list
        List of source csv files.
    image_downsample_size : tuple
        Size to downsample the images to (height, width, depth) or (height, width) for 2D
        images.
    modality_methods : dict
        Dictionary of methods for loading the different modalities.
    fusion_model : class
        Fusion model class. e.g. "TabularCrossmodalAttention".
    test_size : float
        Fraction of data to use for testing (default 0.2).
    graph_creation_method : class
        Graph creation method class.
    graph_maker_instance : graph maker class
        Graph maker class instance.
    layer_mods : dict
        Dictionary of layer modifications to make to the graph maker method.
    dataset : tensor
        Tensor of predictive features. Created in prepare_data().
    data_dims : list
        List of data dimensions [mod1_dim, mod2_dim, img_dim]. Created in prepare_data().
    train_idxs : list
        List of indices for training. Created in setup().
    test_idxs : list
        List of indices for testing. Created in setup().
    graph_data : graph data structure
        Graph data structure. Created in setup().
    own_test_indices : list
        List of indices to use for testing (default None). If None, the test indices are
        randomly selected using the test_size parameter.
    """

    def __init__(
            self,
            fusion_model,
            sources,
            graph_creation_method,
            test_size,
            image_downsample_size=None,
            layer_mods=None,
            extra_log_string_dict=None,
            own_test_indices=None,
    ):
        """
        Parameters
        ----------

        fusion_model : class
            Fusion model class. e.g. "TabularCrossmodalAttention".
        sources : list
            List of source csv files.
        graph_creation_method : class
            Graph creation method class.
        test_size : float
            Fraction of data to use for testing (default 0.2).
        image_downsample_size : tuple
            Size to downsample the images to (height, width, depth) or (height, width) for 2D
            images. None if not downsampling. (default None)
        layer_mods : dict
            Dictionary of layer modifications to make to the graph maker method.
            (default None)
        extra_log_string_dict : dict
            Dictionary of extra strings to add to the log.
        own_test_indices : list
            List of indices to use for testing (default None). If None, the test indices are
            randomly selected using the test_size parameter.

        """

        super().__init__()

        self.sources = sources
        self.image_downsample_size = image_downsample_size
        self.extra_log_string_dict = extra_log_string_dict
        self.modality_methods = {
            "tabular1": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular1,
            "tabular2": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular2,
            "img": LoadDatasets(self.sources, self.image_downsample_size).load_img,
            "tabular_tabular": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular_tabular,
            "tabular_image": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tab_and_img,
        }
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.test_size = test_size
        self.graph_creation_method = graph_creation_method
        self.layer_mods = layer_mods
        self.own_test_indices = own_test_indices

    def prepare_data(self):
        """
        Loads the data with LoadDatasets class

        Returns
        ------
        dataset : tensor
            Tensor of predictive features.
        data_dims : list
            List of data dimensions [mod1_dim, mod2_dim, img_dim]
            i.e. [None, None, [100, 100, 100]] for image only (image dimensions 100 x 100 x 100)
            i.e. [8, 32, None] for tabular1 and tabular2 (tabular1 has 8 features, tabular2 has
            32 features), and no image
        """
        self.dataset, self.data_dims = self.modality_methods[self.modality_type]()

    def setup(self):
        """
        Gets random train and test indices, and gets the graph data structure.

        Returns
        ------
        None
        """
        # get random train and test idxs
        if self.own_test_indices is None:
            [train_dataset, test_dataset] = torch.utils.data.random_split(
                self.dataset, [1 - self.test_size, self.test_size]
            )
            self.train_idxs = train_dataset.indices
            self.test_idxs = test_dataset.indices
        else:
            self.test_idxs = self.own_test_indices
            self.train_idxs = list(
                set(range(len(self.dataset))) - set(self.test_idxs)
            )

        # get the graph data structure
        self.graph_maker_instance = self.graph_creation_method(self.dataset)
        if self.layer_mods is not None:
            # modify the graph maker architecture if specified
            self.graph_maker_instance = model_modifier.modify_model_architecture(
                self.graph_maker_instance,
                self.layer_mods,
            )

        self.graph_data = self.graph_maker_instance.make_graph()

    def get_lightning_module(self):
        """
        Gets the lightning module using the pytorch geometric lightning module for converting
        the graph data structure into a pytorch dataloader.

        Returns
        ------
        lightning_module : lightning module
            Lightning module for converting the graph data structure into a pytorch dataloader.
        """

        lightning_module = LightningNodeData(
            data=self.graph_data,
            input_train_nodes=self.train_idxs,
            input_val_nodes=self.test_idxs,
            input_test_nodes=self.test_idxs,
            input_pred_nodes=self.test_idxs,
            loader="full",
        )

        return lightning_module


class KFoldGraphDataModule:
    """
    Custom pytorch lightning datamodule class for the different modalities with graph data
    structure and k-fold cross validation

    Attributes
    ----------
    num_folds : int
        Total number of folds.
    image_downsample_size : tuple
        Size to downsample the images to (height, width, depth) or (height, width) for 2D
        images.
    sources : list
        List of source csv files. [Tabular1, Tabular2, Image]
    modality_methods : dict
        Dictionary of methods for loading the different modalities.
    fusion_model : class
        Fusion model class. e.g. "TabularCrossmodalAttention".
    graph_creation_method : class
        Graph creation method class.
    graph_maker_instance : graph maker class
        Graph maker class instance.
    layer_mods : dict
        Dictionary of layer modifications to make to the graph maker method.
    dataset : tensor
        Tensor of predictive features.
    data_dims : list
        List of data dimensions [mod1_dim, mod2_dim, img_dim]
    folds : list
        List of tuples of (graph_data, train_idxs, test_idxs)


    """

    def __init__(
            self,
            num_folds,
            fusion_model,
            sources,
            graph_creation_method,
            image_downsample_size=None,
            layer_mods=None,
            extra_log_string_dict=None,
            own_kfold_indices=None,
    ):
        """
        Parameters
        ----------
        num_folds : int
            Total number of folds.
        fusion_model : class
            Fusion model class. e.g. "TabularCrossmodalAttention".
        sources : list
            List of source csv files.
        graph_creation_method : class
            Graph creation method class.
        image_downsample_size : tuple
            Size to downsample the images to (height, width, depth) or (height, width) for 2D
            images. None if not downsampling. (default None)
        layer_mods : dict
            Dictionary of layer modifications to make to the graph maker method.
            (default None)
        extra_log_string_dict : dict
            Dictionary of extra strings to add to the log.
        own_kfold_indices : list
            List of indices to use for k-fold cross validation (default None). If None, the k-fold
            indices are randomly selected. Structure is a list of tuples of (train_indices,
            test_indices). Must be the same length as num_folds.
        """
        super().__init__()
        self.num_folds = num_folds  # total number of folds
        self.image_downsample_size = image_downsample_size
        self.sources = sources
        self.extra_log_string_dict = extra_log_string_dict
        self.modality_methods = {
            "tabular1": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular1,
            "tabular2": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular2,
            "img": LoadDatasets(self.sources, self.image_downsample_size).load_img,
            "tabular_tabular": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular_tabular,
            "tabular_image": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tab_and_img,
        }
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.graph_creation_method = graph_creation_method
        self.layer_mods = layer_mods
        self.own_kfold_indices = own_kfold_indices

    def prepare_data(self):
        """
        Loads the data with LoadDatasets class

        Returns
        ------
        None
        """
        self.dataset, self.data_dims = self.modality_methods[self.modality_type]()

    def kfold_split(self):
        """
        Splits the dataset into k folds

        Returns
        ------
        folds : list
            List of tuples of (train_dataset, test_dataset)
        """
        # get the indices of the dataset
        indices = list(range(len(self.dataset)))

        # split the dataset into k folds
        if self.own_kfold_indices is None:
            kf = KFold(n_splits=self.num_folds, shuffle=True)
            split_kf = kf.split(indices)
        else:
            split_kf = self.own_kfold_indices

        folds = []
        for train_indices, val_indices in split_kf:
            # split the dataset into train and test sets for each fold
            train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            test_dataset = torch.utils.data.Subset(self.dataset, val_indices)

            # append the train and test datasets to the folds list
            folds.append((train_dataset, test_dataset))

        return folds  # list of tuples of (train_dataset, test_dataset)

    def setup(self):
        """
        Gets random train and test indices, and gets the graph data structure.

        Returns
        ------
        None
        """
        self.folds = self.kfold_split()  # get the k folds from kfold_split() function

        new_folds = []
        for fold in self.folds:
            # get the train and test datasets for each fold
            train_dataset, test_dataset = fold
            train_idxs = train_dataset.indices  # get train node idxs from kfold_split()
            test_idxs = test_dataset.indices  # get test node idxs from kfold_split()

            # get the graph data structure
            self.graph_maker_instance = self.graph_creation_method(self.dataset)

            # modify the graph maker architecture if specified
            if self.layer_mods is not None:
                graph_maker = model_modifier.modify_model_architecture(
                    self.graph_maker_instance,
                    self.layer_mods,
                )

            # make the graph data structure
            graph_data = self.graph_maker_instance.make_graph()

            new_folds.append((graph_data, train_idxs, test_idxs))

        self.folds = new_folds  # list of tuples of (graph_data, train_idxs, test_idxs)

    def get_lightning_module(self):
        """
        Returns the lightning module using the pytorch geometric lightning module for converting
        the graph data structure into a pytorch dataloader.

        Returns
        ------
        lightning_modules : list
            List of lightning modules for each fold.

        """
        # get the normal lightning module using the pytorch geometric lightning module

        lightning_modules = []

        for fold in self.folds:
            graph_data, train_idxs, test_idxs = fold

            lightning_module = LightningNodeData(
                data=graph_data,
                input_train_nodes=train_idxs,
                input_val_nodes=test_idxs,
                input_test_nodes=test_idxs,
                input_pred_nodes=test_idxs,
                loader="full",
            )

            lightning_modules.append(lightning_module)

        return lightning_modules  # list of lightning modules for each fold


def prepare_fusion_data(
        prediction_task,
        fusion_model,
        data_paths,
        output_paths,
        kfold=False,
        num_folds=None,
        test_size=0.2,
        batch_size=8,
        multiclass_dimensions=None,
        image_downsample_size=None,
        layer_mods=None,
        max_epochs=1000,
        checkpoint_path=None,
        extra_log_string_dict=None,
        own_early_stopping_callback=None,
        num_workers=0,
        test_indices=None,
        own_kfold_indices=None,
        **kwargs,
):
    """
    Gets the data module for a specific fusion model and training protocol.

    Parameters
    ----------

    prediction_task : str
        Prediction task (binary, multiclass, regression).
    fusion_model : class
        Fusion model class.
    data_paths : dict
        Dictionary of data paths with keys "tabular1", "tabular2", "image".
    output_paths : dict
        Dictionary of output paths with keys "checkpoints", "figures", "losses".
    kfold : bool
        Whether to use kfold cross validation (default False means train/test split).
    num_folds : int or None
        Number of folds for kfold cross validation (default None).
    test_size : float
        Fraction of data to use for testing when using train/test split (default 0.2).
    batch_size : int
        Batch size (default 8).
    multiclass_dimensions : int
        Number of classes for multiclass prediction (default None).
    image_downsample_size : tuple
        Tuple of image dimensions to downsample to (default None).
        e.g. (100, 100, 100) for 3D images, (100, 100) for 2D images.
    layer_mods : dict
        Dictionary of layer modifications (default None).
    max_epochs : int
        Maximum number of epochs to train subspace methods for. (default 1000)
    checkpoint_path : list
        List containing paths to call checkpoint file. Length of the list is the number of trainable subspace models
        in the fusion model (e.g., DAETabImgMaps requires two models to be pre-trained, so we'd pass 2 checkpoint
        paths in the list. (default None will result in the default lightning format).
    extra_log_string_dict : dict
        Dictionary of extra strings to add to a subspace method checkpoint file name (default None).
        e.g. if you're running the same model with different hyperparameters, you can add the hyperparameters.
        Input format {"name": "value"}. In the run name, the extra string will be added
        as "name_value". And a tag will be added as "name_value".
        Default None.
    own_early_stopping_callback : pytorch_lightning.callbacks.EarlyStopping
        Early stopping callback class (default None).
    num_workers : int
        Number of workers for the dataloader (default 0).
    test_indices : list or None
        List of indices to use for testing (default None). If None, then random split is used.
    own_kfold_indices : list or None
        List of indices to use for k-fold cross validation (default None). If None, then random split is used.
    **kwargs : dict
        Extra keyword arguments. Usable for extra arguments for the subspace method MCVAE's early stopping callback: "mcvae_patience" and "mcvae_tolerance".


    Returns
    -------
    dm : datamodule
        Datamodule for the specified fusion method.
    """

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if kfold and own_early_stopping_callback is not None:
        raise ValueError(
            "Cannot use own early stopping callback with kfold cross validation yet. Working on fixing this currently (Nov 2023)")

    # Getting the data paths from the data_paths dictionary into a list
    data_sources = [
        data_paths["tabular1"],
        data_paths["tabular2"],
        data_paths["image"],
    ]

    if not hasattr(fusion_model, "subspace_method"):
        fusion_model.subspace_method = None

    if fusion_model.fusion_type == "graph":
        if kfold:
            graph_data_module = KFoldGraphDataModule(
                num_folds=num_folds,
                fusion_model=fusion_model,
                sources=data_sources,
                graph_creation_method=fusion_model.graph_maker,
                image_downsample_size=image_downsample_size,
                layer_mods=layer_mods,
                extra_log_string_dict=extra_log_string_dict,
                # here is where the kfold split will go
            )
        else:
            graph_data_module = TrainTestGraphDataModule(
                fusion_model,
                sources=data_sources,
                graph_creation_method=fusion_model.graph_maker,
                test_size=test_size,
                image_downsample_size=image_downsample_size,
                layer_mods=layer_mods,
                extra_log_string_dict=extra_log_string_dict,
                own_test_indices=test_indices,
            )

        graph_data_module.prepare_data()
        graph_data_module.setup()
        data_module = graph_data_module.get_lightning_module()

        if kfold:
            # if kfold, then we have a list of lightning modules
            # so we need to set the data dimensions for each lightning module
            for dm_instance in data_module:
                dm_instance.data_dims = graph_data_module.data_dims
                dm_instance.own_early_stopping_callback = own_early_stopping_callback
                dm_instance.graph_maker_instance = graph_data_module.graph_maker_instance
                dm_instance.output_paths = output_paths
                dm_instance.num_folds = num_folds
                dm_instance.prediction_task = prediction_task
                dm_instance.multiclass_dimensions = multiclass_dimensions
        else:
            data_module.data_dims = graph_data_module.data_dims
            data_module.own_early_stopping_callback = own_early_stopping_callback
            data_module.graph_maker_instance = graph_data_module.graph_maker_instance
            data_module.output_paths = output_paths
            data_module.prediction_task = prediction_task
            data_module.multiclass_dimensions = multiclass_dimensions

    else:
        # another other than graph fusion
        if kfold:
            data_module = KFoldDataModule(
                fusion_model,
                sources=data_sources,
                output_paths=output_paths,
                prediction_task=prediction_task,
                batch_size=batch_size,
                num_folds=num_folds,
                multiclass_dimensions=multiclass_dimensions,
                subspace_method=fusion_model.subspace_method,
                image_downsample_size=image_downsample_size,
                layer_mods=layer_mods,
                max_epochs=max_epochs,
                extra_log_string_dict=extra_log_string_dict,
                own_early_stopping_callback=own_early_stopping_callback,
                num_workers=num_workers,
                own_kfold_indices=own_kfold_indices,
                kwargs=kwargs,
            )
        else:
            data_module = TrainTestDataModule(
                fusion_model,
                sources=data_sources,
                output_paths=output_paths,
                prediction_task=prediction_task,
                batch_size=batch_size,
                test_size=test_size,
                multiclass_dimensions=multiclass_dimensions,
                subspace_method=fusion_model.subspace_method,
                image_downsample_size=image_downsample_size,
                layer_mods=layer_mods,
                max_epochs=max_epochs,
                extra_log_string_dict=extra_log_string_dict,
                own_early_stopping_callback=own_early_stopping_callback,
                num_workers=num_workers,
                test_indices=test_indices,
                kwargs=kwargs,
            )
        data_module.prepare_data()
        data_module.setup(checkpoint_path=checkpoint_path)

    return data_module
