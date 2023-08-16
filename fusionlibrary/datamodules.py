"""
This file contains the datamodules for the different modalities and the different
fusion methods. The datamodules are used to load the data and create the
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
from fusionlibrary.eval_functions import plot_graph


def downsample_img_batch(imgs, output_size):
    """
    Downsamples a batch of images to a specified size.

    Parameters
    ----------
    imgs : array-like
        Batch of images.
    output_size : tuple
        Size to downsample the images to (height, width).

    Returns
    -------
    downsampled_img : array-like
        Downsampled image.
    """
    imgs = torch.unsqueeze(imgs, dim=0)
    downsampled_img = F.interpolate(imgs, size=output_size, mode="nearest")

    return torch.squeeze(downsampled_img)


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

    def __init__(self, sources):
        """
        Parameters
        ----------
        sources : list
            List of source csv files.
            [tabular1_source, tabular2_source, img_source]

        Raises
        ------
        ValueError
            If sources is not a list.
        """
        self.tabular1_source, self.tabular2_source, self.img_source = sources
        self.image_downsample_size = (100, 100, 100)  # can choose own image size here

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

        img_dim = list(all_scans_ds.shape[1:])

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
        tab1_df = pd.read_csv(self.tabular1_source).set_index("study_id")
        tab2_df = pd.read_csv(self.tabular2_source).set_index("study_id")

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

        tab1_df = pd.read_csv(self.tabular1_source).set_index("study_id")
        tab1_features = torch.Tensor(tab1_df.drop(columns=["pred_label"]).values)
        label_df = tab1_df[["pred_label"]]

        imgs = torch.load(self.img_source)
        imgs = downsample_img_batch(imgs, self.image_downsample_size)

        self.dataset = CustomDataset([tab1_features, imgs], label_df)
        mod1_dim = tab1_features.shape[1]
        img_dim = list(imgs.shape[1:])

        return self.dataset, [mod1_dim, None, img_dim]


class CustomDataModule(pl.LightningDataModule):
    """
    Custom pytorch lightning datamodule class for the different modalities.

    Attributes
    ----------
    sources : list
        List of source csv files.
    modality_type : str
        Type of modality (tabular1, tabular2, img, both_tab, tab_img).
    batch_size : int
        Batch size (default 8).
    subspace_method : class
        Subspace method class (default None) (only for subspace methods).

    Methods
    -------
    prepare_data()
        Loads the data with LoadDatasets class.
    setup()
        Splits the data into train and test sets, and runs the subspace method if specified.
    train_dataloader()
        Returns the dataloader for training.
    val_dataloader()
        Returns the dataloader for validation.
    train_eval_dataloader()
        Returns the dataloader for evaluation.
    """

    def __init__(
        self,
        params,
        modality_type,
        sources,
        batch_size=8,
        subspace_method=None,
    ):
        """
        Parameters
        ----------

        params : dict
            Dictionary of parameters.
        modality_type : str
            Type of modality (tabular1, tabular2, img, both_tab, tab_img).
        sources : list
            List of source csv files.
        batch_size : int
            Batch size (default 8).
        subspace_method : class
            Subspace method class (default None) (only for subspace methods).

        Raises
        ------
        ValueError
            If modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.
        """
        super().__init__()

        self.sources = sources

        self.modality_methods = {
            "tabular1": LoadDatasets(self.sources).load_tabular1,
            "tabular2": LoadDatasets(self.sources).load_tabular2,
            "img": LoadDatasets(self.sources).load_img,
            "both_tab": LoadDatasets(self.sources).load_both_tabular,
            "tab_img": LoadDatasets(self.sources).load_tab_and_img,
        }

        self.modality_type = modality_type
        self.batch_size = batch_size
        self.test_size = params["test_size"]
        self.num_latent_dims = params["subspace_latdims"]
        self.subspace_method = subspace_method

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
    ):
        """
        Splits the data into train and test sets, and runs the subspace method if specified

        Returns
        ------
        train_dataloader : dataloader
            Dataloader for training.
        val_dataloader : dataloader
            Dataloader for validation.
        train_eval_dataloader : dataloader
            Dataloader for evaluation.
        """
        [self.train_dataset, self.test_dataset] = torch.utils.data.random_split(
            self.dataset, [1 - self.test_size, self.test_size]
        )

        if self.subspace_method is not None:  # if subspace method is specified
            subspace_method = self.subspace_method(self)
            train_latents, train_labels = subspace_method.train(
                self.train_dataset, self.test_dataset
            )
            test_latents, test_labels, data_dims = subspace_method.convert_to_latent(
                self.test_dataset
            )

            # make a new CustomDataset with the latent features
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

    def train_eval_dataloader(self):
        """
        Returns the dataloader for evaluation.

        Returns
        -------
        dataloader : dataloader
            Dataloader for evaluation.
        """
        return DataLoader(
            self.train_dataset, batch_size=1, shuffle=False, num_workers=0
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
    modality_type : str
        Type of modality (tabular1, tabular2, img, both_tab, tab_img).
    batch_size : int
        Batch size (default 8).
    subspace_method : class
        Subspace method class (default None) (only for subspace methods).

    Methods
    -------
    prepare_data()
        Loads the data with LoadDatasets class.
    kfold_split()
        Splits the dataset into k folds.
    setup()
        Splits the data into train and test sets, and runs the subspace method if specified.
    train_dataloader()
        Returns the dataloader for training.
    val_dataloader()
        Returns the dataloader for validation.
    train_eval_dataloader()
        Returns the dataloader for evaluation.
    """

    def __init__(
        self, params, modality_type, sources, batch_size=8, subspace_method=None
    ):
        """
        Parameters
        ----------

        params : dict
            Dictionary of parameters.
        modality_type : str
            Type of modality (tabular1, tabular2, img, both_tab, tab_img).
        sources : list
            List of source csv files.
        batch_size : int
            Batch size (default 8).
        subspace_method : class
            Subspace method class (default None) (only for subspace methods).

        Raises
        ------
        ValueError
            If modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.

        """
        super().__init__()

        self.num_folds = params["num_k"]  # total number of folds
        self.sources = sources
        self.modality_methods = {
            "tabular1": LoadDatasets(self.sources).load_tabular1,
            "tabular2": LoadDatasets(self.sources).load_tabular2,
            "img": LoadDatasets(self.sources).load_img,
            "both_tab": LoadDatasets(self.sources).load_both_tabular,
            "tab_img": LoadDatasets(self.sources).load_tab_and_img,
        }
        self.modality_type = modality_type
        self.batch_size = batch_size
        self.num_latent_dims = params["subspace_latdims"]
        self.subspace_method = (
            subspace_method  # subspace method class (only for subspace methods)
        )

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
    ):
        """
        Splits the data into train and test sets, and runs the subspace method if specified

        Returns
        ------
        train_dataloader : dataloader
            Dataloader for training.
        val_dataloader : dataloader
            Dataloader for validation.
        train_eval_dataloader : dataloader
            Dataloader for evaluation.
        """
        self.folds = self.kfold_split()  # get the k folds from kfold_split() function

        # if subspace method is specified, run the subspace method on each fold
        if self.subspace_method is not None:
            new_folds = []
            for fold in self.folds:
                train_dataset, test_dataset = fold
                subspace_method = self.subspace_method(self)
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

    def train_eval_dataloader(self, fold_idx):
        """
        Returns the dataloader for evaluation.

        Parameters
        ----------
        fold_idx : int
            Index of the fold to use.

        Returns
        -------
        dataloader : dataloader
            Dataloader for evaluation.
        """
        self.train_dataset, self.test_dataset = self.folds[fold_idx]

        return DataLoader(
            self.train_dataset, batch_size=1, shuffle=False, num_workers=0
        )


class GraphDataModule:
    """
    Custom pytorch lightning datamodule class for the different modalities with graph data
    structure.

    Attributes
    ----------
    sources : list
        List of source csv files.
    modality_type : str
        Type of modality (tabular1, tabular2, img, both_tab, tab_img).
    batch_size : int
        Batch size (default 8) (although not used for graph methods).
    graph_creation_method : class
        Graph creation method class.

    Methods
    -------
    prepare_data()
        Loads the data with LoadDatasets class.
    setup()
        Gets random train and test indices, and gets the graph data structure.
    get_lightning_module()
        Returns the lightning module using the pytorch geometric lightning module for converting
        the graph data structure into a pytorch dataloader.
    """

    def __init__(
        self,
        params,
        modality_type,
        sources,
        graph_creation_method,
        batch_size=8,
    ):
        """
        Parameters
        ----------

        params : dict
            Dictionary of parameters.
        modality_type : str
            Type of modality (tabular1, tabular2, img, both_tab, tab_img).
        sources : list
            List of source csv files.
        graph_creation_method : class
            Graph creation method class.
        batch_size : int
            Batch size (default 8) (although not used for graph methods).

        Raises
        ------
        ValueError
            If modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.
        """

        super().__init__()

        self.sources = sources
        self.modality_methods = {
            "tabular1": LoadDatasets(self.sources).load_tabular1,
            "tabular2": LoadDatasets(self.sources).load_tabular2,
            "img": LoadDatasets(self.sources).load_img,
            "both_tab": LoadDatasets(self.sources).load_both_tabular,
            "tab_img": LoadDatasets(self.sources).load_tab_and_img,
        }
        self.modality_type = modality_type
        self.batch_size = batch_size
        self.test_size = params["test_size"]
        self.graph_creation_method = graph_creation_method
        self.params = params

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
        """
        # get random train and test idxs
        [self.train_dataset, self.test_dataset] = torch.utils.data.random_split(
            self.dataset, [1 - self.test_size, self.test_size]
        )
        self.train_idxs = self.train_dataset.indices
        self.test_idxs = self.test_dataset.indices

        # get the graph data structure
        self.graph_data = self.graph_creation_method(self.dataset)

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
    sources : list
        List of source csv files.
    modality_type : str
        Type of modality (tabular1, tabular2, img, both_tab, tab_img).
    batch_size : int
        Batch size (default 8) (although not used for graph methods).
    graph_creation_method : class
        Graph creation method class.

    Methods
    -------
    prepare_data()
        Loads the data with LoadDatasets class.
    kfold_split()
        Splits the dataset into k folds.
    setup()
        Gets random train and test indices, and gets the graph data structure.
    get_lightning_module()
        Returns the lightning module using the pytorch geometric lightning module for converting
        the graph data structure into a pytorch dataloader.

    """

    def __init__(
        self,
        params,
        modality_type,
        sources,
        graph_creation_method,
        batch_size=8,
    ):
        """
        Parameters
        ----------
        params : dict
            Dictionary of parameters.
        modality_type : str
            Type of modality (tabular1, tabular2, img, both_tab, tab_img).
        sources : list
            List of source csv files.
        graph_creation_method : class
            Graph creation method class.
        batch_size : int
            Batch size (default 8) (although not used for graph methods).

        Raises
        ------
        ValueError
            If modality_type is not one of the following: tabular1, tabular2, img, both_tab,
            tab_img.
        """
        super().__init__()
        self.num_folds = params["num_k"]  # total number of folds
        self.sources = sources
        self.modality_methods = {
            "tabular1": LoadDatasets(self.sources).load_tabular1,
            "tabular2": LoadDatasets(self.sources).load_tabular2,
            "img": LoadDatasets(self.sources).load_img,
            "both_tab": LoadDatasets(self.sources).load_both_tabular,
            "tab_img": LoadDatasets(self.sources).load_tab_and_img,
        }
        self.modality_type = modality_type
        self.batch_size = batch_size
        self.graph_creation_method = graph_creation_method
        self.params = params

    def prepare_data(self):
        """
        Loads the data with LoadDatasets class
        """
        self.dataset, self.data_dims = self.modality_methods[self.modality_type]()

    def kfold_split(self):
        """
        Splits the dataset into k folds
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
        """
        self.folds = self.kfold_split()  # get the k folds from kfold_split() function

        new_folds = []
        for fold in self.folds:
            train_dataset, test_dataset = fold

            train_idxs = train_dataset.indices  # get train node idxs from kfold_split()
            test_idxs = test_dataset.indices  # get test node idxs from kfold_split()

            # get the graph data structure
            graph_data = self.graph_creation_method(self.dataset)

            # plot and save the graph?
            # TODO remove?
            plot_graph(graph_data, self.params)

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


def get_data_module(init_model, params):
    # needs model attributes: fusion_type and modality_type
    # needs params: kfold, pred_type etc
    if init_model.fusion_type == "graph":
        if params["kfold_flag"]:
            dmg = KFoldGraphDataModule(
                params,
                init_model.modality_type,
                sources=params["data_sources"],
                graph_creation_method=init_model.graph_maker,
            )
        else:
            dmg = GraphDataModule(
                params,
                init_model.modality_type,
                sources=params["data_sources"],
                graph_creation_method=init_model.graph_maker,
            )

        dmg.prepare_data()
        dmg.setup()
        dm = dmg.get_lightning_module()

        if params["kfold_flag"]:
            for dm_instance in dm:
                dm_instance.data_dims = dmg.data_dims
        else:
            dm.data_dims = dmg.data_dims

    else:
        # another other than graph fusion
        if params["kfold_flag"]:
            datamodule_func = KFoldDataModule
        else:
            datamodule_func = CustomDataModule

        dm = datamodule_func(
            params,
            init_model.modality_type,
            sources=params["data_sources"],
            subspace_method=init_model.subspace_method,
        )
        dm.prepare_data()
        dm.setup()

    return dm
