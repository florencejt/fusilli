"""
This file contains the data for the different modalities and the different
fusion methods. The data are used to load the data and create the
dataloader objects for training and validation.
"""

# imports
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data.lightning import LightningNodeData
from fusilli.utils import model_modifier


# from fusilli.eval import plot_graph


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
            f"output_size must be smaller than image dimensions, but got {output_size} and image dimensions {imgs.shape[2:]}"
        )

    downsampled_img = F.interpolate(imgs, size=output_size, mode="nearest")

    return downsampled_img


class CustomDataset(Dataset):
    """
    Custom dataset class for multimodal data.

    Attributes
    ----------
    pred_features : list or tensor
        List of tensors or tensor of predictive features
        (i.e. tabular or image data without labels).
    labels : dataframe
        Dataframe of labels (column name must be "pred_label").

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Returns the item at the specified index.

    """

    def __init__(self, pred_features, labels):
        """
        Parameters
        ----------

        pred_features : list or tensor
            List of tensors or tensor of predictive features
            (i.e. tabular or image data without labels).
        labels : dataframe
            Dataframe of labels (column name must be "pred_label").

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
        label_type = labels[["pred_label"]].values.dtype
        self.labels = torch.tensor(labels[["pred_label"]].to_numpy().reshape(-1))
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
        Size to downsample the images to (height, width, depth).

    Methods
    -------
    load_tabular1()
        Loads the tabular1-only dataset.
    load_tabular2()
        Loads the tabular2-only dataset.
    load_img()
        Loads the image-only dataset.
    load_both_tabular()
        Loads the tabular1 and tabular2 multimodal dataset.
    load_tab_and_img()
        Loads the tabular1 and image multimodal dataset.
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
        """
        self.tabular1_source, self.tabular2_source, self.img_source = sources
        self.image_downsample_size = (
            img_downsample_dims  # can choose own image size here
        )

        # read in the csv files and raise errors if they don't have the right columns
        # or if the index column is not named "study_id"
        tab1_df = pd.read_csv(self.tabular1_source)
        tab2_df = pd.read_csv(self.tabular2_source)

        if "study_id" not in tab1_df.columns:
            raise ValueError("The CSV must have an index column named 'study_id'.")
        if "pred_label" not in tab1_df.columns:
            raise ValueError("The CSV must have a label column named 'pred_label'.")
        if "study_id" not in tab2_df.columns:
            raise ValueError("The CSV must have an index column named 'study_id'.")
        if "pred_label" not in tab2_df.columns:
            raise ValueError("The CSV must have a label column named 'pred_label'.")

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

        tab_df.set_index("study_id", inplace=True)

        pred_features = torch.Tensor(tab_df.drop(columns=["pred_label"]).values)
        pred_label = tab_df[["pred_label"]]

        self.dataset = CustomDataset(pred_features, pred_label)

        mod1_dim = pred_features.shape[1]

        return self.dataset, [mod1_dim, None, None]

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

        tab_df.set_index("study_id", inplace=True)

        pred_features = torch.Tensor(tab_df.drop(columns=["pred_label"]).values)
        pred_label = tab_df[["pred_label"]]

        self.dataset = CustomDataset(pred_features, pred_label)
        mod2_dim = pred_features.shape[1]

        return self.dataset, [None, mod2_dim, None]

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

        label_df.set_index("study_id", inplace=True)

        pred_label = label_df[["pred_label"]]

        self.dataset = CustomDataset(all_scans_ds, pred_label)

        img_dim = list(all_scans_ds.shape[2:])  # not including batch size or channels

        return self.dataset, [None, None, img_dim]

    def load_both_tabular(self):
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

        tab1_df.set_index("study_id", inplace=True)
        tab2_df.set_index("study_id", inplace=True)

        tab1_pred_features = torch.Tensor(tab1_df.drop(columns=["pred_label"]).values)
        tab2_pred_features = torch.Tensor(tab2_df.drop(columns=["pred_label"]).values)

        pred_label = tab1_df[["pred_label"]]
        self.dataset = CustomDataset(
            [tab1_pred_features, tab2_pred_features], pred_label
        )

        mod1_dim = tab1_pred_features.shape[1]
        mod2_dim = tab2_pred_features.shape[1]

        return self.dataset, [mod1_dim, mod2_dim, None]

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

        tab1_df.set_index("study_id", inplace=True)

        tab1_features = torch.Tensor(tab1_df.drop(columns=["pred_label"]).values)
        label_df = tab1_df[["pred_label"]]

        imgs = torch.load(self.img_source)
        imgs = downsample_img_batch(imgs, self.image_downsample_size)

        self.dataset = CustomDataset([tab1_features, imgs], label_df)
        mod1_dim = tab1_features.shape[1]
        img_dim = list(imgs.shape[2:])  # not including batch size or channels

        return self.dataset, [mod1_dim, None, img_dim]


class CustomDataModule(pl.LightningDataModule):
    """
    Custom pytorch lightning datamodule class for the different modalities.

    Attributes
    ----------
    sources : list
        List of source csv files.
    image_downsample_size : tuple
        Size to downsample the images to (height, width, depth) or (height, width) for 2D
        images.
    modality_methods : dict
        Dictionary of methods for loading the different modalities.
    fusion_model : str
        fusion model class. e.g. "TabularCrossmodalAttention".
    batch_size : int
        Batch size (default 8).
    test_size : float
        Fraction of data to use for testing (default 0.2).
    pred_type : str
        Prediction type (binary, multiclass, regression).
    multiclass_dims : int
        Number of classes for multiclass prediction (default None).
    subspace_method : class
        Subspace method class (default None) (only for subspace methods).
    layer_mods : dict
        Dictionary of layer modifications to make to the subspace method.
        (default None)
    max_epochs : int
        Maximum number of epochs to train subspace methods for. (default 1000)
    dataset : tensor
        Tensor of predictive features.
    data_dims : list
        List of data dimensions [mod1_dim, mod2_dim, img_dim]
    train_dataset : tensor
        Tensor of predictive features for training.
    test_dataset : tensor
        Tensor of predictive features for testing.

    """

    def __init__(
        self,
        params,
        fusion_model,
        sources,
        batch_size,
        subspace_method=None,
        image_downsample_size=None,
        layer_mods=None,
        max_epochs=1000,
        extra_log_string_dict=None,
        own_early_stopping_callback=None,
    ):
        """
        Parameters
        ----------

        params : dict
            Dictionary of parameters.
        fusion_model : str
            Fusion model class. e.g. "TabularCrossmodalAttention".
        sources : list
            List of source csv files.
        batch_size : int
            Batch size (default 8).
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


        Raises
        ------
        ValueError
            If fusion_model.modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.
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
            "both_tab": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_both_tabular,
            "tab_img": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tab_and_img,
        }
        self.params = params
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.batch_size = batch_size
        self.test_size = params["test_size"]
        self.pred_type = params["pred_type"]
        if self.pred_type == "multiclass":
            self.multiclass_dims = params["multiclass_dims"]
        else:
            self.multiclass_dims = None
        self.subspace_method = subspace_method
        self.layer_mods = layer_mods
        self.max_epochs = max_epochs
        self.own_early_stopping_callback = own_early_stopping_callback

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
        # TODO add in some evaluation figures of the subspace methods
        self.dataset, self.data_dims = self.modality_methods[self.modality_type]()

    def setup(
        self,
        checkpoint_path=None,
    ):
        """
        Splits the data into train and test sets, and runs the subspace method if specified

        Returns
        ------
        train_dataloader : dataloader
            Dataloader for training.
        val_dataloader : dataloader
            Dataloader for validation.
        """
        [self.train_dataset, self.test_dataset] = torch.utils.data.random_split(
            self.dataset, [1 - self.test_size, self.test_size]
        )

        if self.subspace_method is not None:  # if subspace method is specified
            if checkpoint_path is None:
                subspace_method = self.subspace_method(
                    self,
                    max_epochs=self.max_epochs,
                    k=None,
                )

                if self.layer_mods is not None:
                    # if subspace method in layer_mods
                    subspace_method = model_modifier.modify_model_architecture(
                        subspace_method,
                        self.layer_mods,
                    )

                train_latents, train_labels = subspace_method.train(
                    self.train_dataset, self.test_dataset
                )

                (
                    test_latents,
                    test_labels,
                    data_dims,
                ) = subspace_method.convert_to_latent(self.test_dataset)

                self.train_dataset = CustomDataset(train_latents, train_labels)
                self.test_dataset = CustomDataset(test_latents, test_labels)
                self.data_dims = data_dims

            else:
                # we have the checkpoint paths for the subspace models

                subspace_method = self.subspace_method(
                    self,
                    max_epochs=self.max_epochs,
                    k=None,
                    checkpoint_path=checkpoint_path,
                )

                (
                    train_latents,
                    train_labels,
                    data_dims,
                ) = subspace_method.convert_to_latent(self.train_dataset)

                (
                    test_latents,
                    test_labels,
                    data_dims,
                ) = subspace_method.convert_to_latent(self.test_dataset)

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
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
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
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
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
        List of source csv files.
    image_downsample_size : tuple
        Size to downsample the images to (height, width, depth) or (height, width) for 2D
        images.
    modality_methods : dict
        Dictionary of methods for loading the different modalities.
    fusion_model : class
        Fusion model class. e.g. "TabularCrossmodalAttention".
    batch_size : int
        Batch size (default 8).
    test_size : float
        Fraction of data to use for testing (default 0.2).
    pred_type : str
        Prediction type (binary, multiclass, regression).
    multiclass_dims : int
        Number of classes for multiclass prediction (default None).
    subspace_method : class
        Subspace method class (default None) (only for subspace methods).
    layer_mods : dict
        Dictionary of layer modifications to make to the subspace method.
        (default None)
    max_epochs : int
        Maximum number of epochs to train subspace methods for. (default 1000)
    dataset : tensor
        Tensor of predictive features.
    data_dims : list
        List of data dimensions [mod1_dim, mod2_dim, img_dim]
    train_dataset : tensor
        Tensor of predictive features for training.
    test_dataset : tensor
        Tensor of predictive features for testing.
    """

    def __init__(
        self,
        params,
        fusion_model,
        sources,
        batch_size,
        subspace_method=None,
        image_downsample_size=None,
        layer_mods=None,
        max_epochs=1000,
        extra_log_string_dict=None,
        own_early_stopping_callback=None,
    ):
        """
        Parameters
        ----------

        params : dict
            Dictionary of parameters.
        fusion_model : class
            Fusion model class. e.g. "TabularCrossmodalAttention".
        sources : list
            List of source csv files.
        batch_size : int
            Batch size (default 8).
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


        Raises
        ------
        ValueError
            If fusion_model.modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.

        """
        super().__init__()

        self.num_folds = params["num_k"]  # total number of folds
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
            "both_tab": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_both_tabular,
            "tab_img": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tab_and_img,
        }
        self.params = params
        self.pred_type = params["pred_type"]
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.batch_size = batch_size
        self.subspace_method = (
            subspace_method  # subspace method class (only for subspace methods)
        )
        if self.pred_type == "multiclass":
            self.multiclass_dims = params["multiclass_dims"]
        else:
            self.multiclass_dims = None
        self.layer_mods = layer_mods
        self.max_epochs = max_epochs
        self.own_early_stopping_callback = own_early_stopping_callback

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
        # splits the dataset into k folds
        kf = KFold(n_splits=self.num_folds, shuffle=True)

        # get the indices of the dataset
        indices = list(range(len(self.dataset)))

        folds = []
        for train_indices, val_indices in kf.split(indices):
            # split the dataset into train and test sets for each fold
            train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            test_dataset = torch.utils.data.Subset(self.dataset, val_indices)
            folds.append((train_dataset, test_dataset))

        return folds  # list of tuples of (train_dataset, test_dataset)

    def setup(
        self,
        checkpoint_path=None,
    ):
        """
        Splits the data into train and test sets, and runs the subspace method if specified

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
            if checkpoint_path is None:
                new_folds = []
                for k, fold in enumerate(self.folds):
                    train_dataset, test_dataset = fold
                    subspace_method = self.subspace_method(
                        self,
                        k=k,
                        max_epochs=self.max_epochs,
                    )

                    if self.layer_mods is not None:
                        # if subspace method in layer_mods
                        subspace_method = model_modifier.modify_model_architecture(
                            subspace_method,
                            self.layer_mods,
                        )

                    train_latents, train_labels = subspace_method.train(
                        train_dataset, test_dataset
                    )
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

            else:
                new_folds = []

                for k, fold in enumerate(self.folds):
                    train_dataset, test_dataset = fold

                    subspace_method = self.subspace_method(
                        self,
                        k=k,
                        max_epochs=self.max_epochs,
                        checkpoint_path=checkpoint_path,
                    )

                    (
                        train_latents,
                        train_labels,
                        data_dims,
                    ) = subspace_method.convert_to_latent(train_dataset)

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
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
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
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )


class GraphDataModule:
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
    params : dict
        Dictionary of parameters.
    layer_mods : dict
        Dictionary of layer modifications to make to the graph maker method.
    dataset : tensor
        Tensor of predictive features.
    data_dims : list
        List of data dimensions [mod1_dim, mod2_dim, img_dim]
    train_dataset : tensor
        Tensor of predictive features for training.
    train_idxs : list
        List of indices for training.
    test_dataset : tensor
        Tensor of predictive features for testing.
    test_idxs : list
        List of indices for testing.
    graph_data : graph data structure
        Graph data structure.
    """

    def __init__(
        self,
        params,
        fusion_model,
        sources,
        graph_creation_method,
        image_downsample_size=None,
        layer_mods=None,
        extra_log_string_dict=None,
    ):
        """
        Parameters
        ----------

        params : dict
            Dictionary of parameters.
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

        Raises
        ------
        ValueError
            If fusion_model.modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.
        """

        super().__init__()

        self.sources = sources
        self.image_downsample_size = image_downsample_size
        self.params = params
        self.extra_log_string_dict = extra_log_string_dict
        self.modality_methods = {
            "tabular1": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular1,
            "tabular2": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular2,
            "img": LoadDatasets(self.sources, self.image_downsample_size).load_img,
            "both_tab": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_both_tabular,
            "tab_img": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tab_and_img,
        }
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.test_size = params["test_size"]
        self.graph_creation_method = graph_creation_method
        self.layer_mods = layer_mods

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
        [self.train_dataset, self.test_dataset] = torch.utils.data.random_split(
            self.dataset, [1 - self.test_size, self.test_size]
        )
        self.train_idxs = self.train_dataset.indices
        self.test_idxs = self.test_dataset.indices

        # get the graph data structure
        graph_maker = self.graph_creation_method(self.dataset)
        if self.layer_mods is not None:
            # if subspace method in layer_mods
            graph_maker = model_modifier.modify_model_architecture(
                graph_maker,
                self.layer_mods,
            )

        self.graph_data = graph_maker.make_graph()
        # self.graph_data = self.graph_creation_method(self.dataset)

        # plot and save the graph
        # TODO move to the soon-to-be-made plotting module? or just remove
        # plot_graph(self.graph_data, self.params)

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
        List of source csv files.
    modality_methods : dict
        Dictionary of methods for loading the different modalities.
    fusion_model : class
        Fusion model class. e.g. "TabularCrossmodalAttention".
    graph_creation_method : class
        Graph creation method class.
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
        params,
        fusion_model,
        sources,
        graph_creation_method,
        image_downsample_size=None,
        layer_mods=None,
        extra_log_string_dict=None,
    ):
        """
        Parameters
        ----------
        params : dict
            Dictionary of parameters.
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
        extra_log_string_dict : dict
            Dictionary of extra strings to add to the log.


        Raises
        ------
        ValueError
            If fusion_model.modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.
        """
        super().__init__()
        self.num_folds = params["num_k"]  # total number of folds
        self.image_downsample_size = image_downsample_size
        self.sources = sources
        self.params = params
        self.extra_log_string_dict = extra_log_string_dict
        self.modality_methods = {
            "tabular1": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular1,
            "tabular2": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tabular2,
            "img": LoadDatasets(self.sources, self.image_downsample_size).load_img,
            "both_tab": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_both_tabular,
            "tab_img": LoadDatasets(
                self.sources, self.image_downsample_size
            ).load_tab_and_img,
        }
        self.fusion_model = fusion_model
        self.modality_type = self.fusion_model.modality_type
        self.graph_creation_method = graph_creation_method
        self.layer_mods = layer_mods

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
            List of tuples of (graph_data, train_idxs, test_idxs)
        """
        # splits the dataset into k folds
        kf = KFold(n_splits=self.num_folds, shuffle=True)
        indices = list(range(len(self.dataset)))  # get the indices of the dataset

        folds = []
        for train_indices, val_indices in kf.split(indices):
            # split the dataset into train and test sets for each fold
            train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            test_dataset = torch.utils.data.Subset(self.dataset, val_indices)
            folds.append(
                (train_dataset, test_dataset)
            )  # list of tuples of (train_dataset, test_dataset)
        return folds

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
            train_dataset, test_dataset = fold

            train_idxs = train_dataset.indices  # get train node idxs from kfold_split()
            test_idxs = test_dataset.indices  # get test node idxs from kfold_split()
            # get the graph data structure
            graph_maker = self.graph_creation_method(self.dataset)
            if self.layer_mods is not None:
                # if subspace method in layer_mods
                graph_maker = model_modifier.modify_model_architecture(
                    graph_maker,
                    self.layer_mods,
                )

            graph_data = graph_maker.make_graph()

            # plot and save the graph?
            # TODO remove?
            # plot_graph(graph_data, self.params)

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


def get_data_module(
    fusion_model,
    params,
    batch_size=8,
    image_downsample_size=None,
    layer_mods=None,
    max_epochs=1000,
    optional_suffix="",
    checkpoint_path=None,
    extra_log_string_dict=None,
    own_early_stopping_callback=None,
):
    """
    Gets the data module for the specified modality and fusion type.

    Parameters
    ----------
    fusion_model : class
        Fusion model class.
    params : dict
        Dictionary of parameters.
    batch_size : int
        Batch size (default 8).
    image_downsample_size : list
        List of image dimensions to downsample to (default None).
    layer_mods : dict
        Dictionary of layer modifications (default None).
    optional_suffix : str
        Optional suffix added to data source file names (default None).
    checkpoint_path : str
        Path to call checkpoint file (default None will result in the default lightning format).
    extra_log_string_dict : dict
        Dictionary of extra strings to add to a subspace method checkpoint file name (default None).
        e.g. if you're running the same model with different hyperparameters, you can add the hyperparameters.
        Input format {"name": "value"}. In the run name, the extra string will be added
        as "name_value". And a tag will be added as "name_value".
        Default None.
    own_early_stopping_callback : pytorch_lightning.callbacks.EarlyStopping
        Early stopping callback class (default None).


    Returns
    -------
    dm : datamodule
        Datamodule for the specified fusion method.
    """

    data_sources = [
        params[f"tabular1_source{optional_suffix}"],
        params[f"tabular2_source{optional_suffix}"],
        params[f"img_source{optional_suffix}"],
    ]

    if not hasattr(fusion_model, "subspace_method"):
        fusion_model.subspace_method = None

    if fusion_model.fusion_type == "graph":
        if params["kfold_flag"]:
            dmg = KFoldGraphDataModule(
                params,
                fusion_model,
                sources=data_sources,
                graph_creation_method=fusion_model.graph_maker,
                image_downsample_size=image_downsample_size,
                layer_mods=layer_mods,
                extra_log_string_dict=extra_log_string_dict,
            )
        else:
            dmg = GraphDataModule(
                params,
                fusion_model,
                sources=data_sources,
                graph_creation_method=fusion_model.graph_maker,
                image_downsample_size=image_downsample_size,
                layer_mods=layer_mods,
                extra_log_string_dict=extra_log_string_dict,
            )

        dmg.prepare_data()
        dmg.setup()
        dm = dmg.get_lightning_module()

        if params["kfold_flag"]:
            for dm_instance in dm:
                dm_instance.data_dims = dmg.data_dims
                dm_instance.own_early_stopping_callback = own_early_stopping_callback
        else:
            dm.data_dims = dmg.data_dims
            dm.own_early_stopping_callback = own_early_stopping_callback

    else:
        # another other than graph fusion
        if params["kfold_flag"]:
            datamodule_func = KFoldDataModule
        else:
            datamodule_func = CustomDataModule

        dm = datamodule_func(
            params,
            fusion_model,
            sources=data_sources,
            subspace_method=fusion_model.subspace_method,
            batch_size=batch_size,
            image_downsample_size=image_downsample_size,
            layer_mods=layer_mods,
            max_epochs=max_epochs,
            extra_log_string_dict=extra_log_string_dict,
            own_early_stopping_callback=own_early_stopping_callback,
        )
        dm.prepare_data()
        dm.setup(checkpoint_path=checkpoint_path)

    return dm
