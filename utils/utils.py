import os
import logging
import copy
import torch
import numpy as np
import math
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import ConcatDataset
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from utils.model import *
from utils.datasets import ImageFolder_SingleDatasetTIF, ParquetDataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def init_nets(n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset == 'RS-5':
        n_classes = 5
    elif args.dataset == 'RS-10':
        n_classes = 10
    elif args.dataset == 'RS-15':
        n_classes = 15
    for net_i in range(n_parties):
        net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, args.dataset)
        if device == 'cpu':
            net.to(device)
        else:
            net = net.cuda()
        nets[net_i] = net
    return nets


def load_parquet_data(datadir, n_parties, dataset, train_distribution, val_distribution):
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if dataset == 'RS-5':
        n_classes = 5
    elif dataset == 'RS-10':
        n_classes = 10
    elif dataset == "RS-15":
        n_classes = 15
    all_X_train = []
    all_y_train = []
    net_dataidx_map = {}
    all_traindata_cls_counts = []

    train_root = f"{datadir}/train.parquet"
    val_root = f"{datadir}/test.parquet"

    if train_distribution == 'noniid-1' and dataset == 'RS-15':
        train_map = f"{datadir}/map/map-NIID1.json"
    elif train_distribution == 'noniid-2' and dataset == 'RS-15':
        train_map = f"{datadir}/map/map-NIID2.json"

    elif train_distribution == 'noniid-1' and dataset == 'RS-5':
        train_map = f"{datadir}/map/map-RS5-NIID1.json"
    elif train_distribution == 'noniid-2' and dataset == 'RS-5':
        train_map = f"{datadir}/map/map-RS5-NIID2.json"

    if dataset == 'RS-15':
        if val_distribution == 1:
            val_map = f"{datadir}/map/map-balanced.json"
        else:
            val_map = f"{datadir}/map/map-imbalanced.json"
    elif dataset == 'RS-5':
        if val_distribution == 1:
            val_map = f"{datadir}/map/map-RS5-balanced.json"
        else:
            val_map = f"{datadir}/map/map-RS5-imbalanced.json"

    dl_obj = ParquetDataset

    test_parquet = f"{datadir}/test.parquet"
    train_parquet = f"{datadir}/train.parquet"

    xray_test_ds = ParquetDataset(test_parquet, map_path=val_map, client_idx=None, transform=transform_test)

    X_test = np.array([xray_test_ds[i][0].numpy() for i in range(len(xray_test_ds))])
    y_test = np.array([xray_test_ds[i][1] for i in range(len(xray_test_ds))])

    for client_idx in range(n_parties):
        # load train set
        xray_train_ds = ParquetDataset(train_parquet, map_path=train_map, client_idx=client_idx,
                                       transform=transform_train)

        # get features and labels
        # X_train = np.array([xray_train_ds[i][0].numpy() for i in range(len(xray_train_ds))])
        # y_train = np.array([xray_train_ds[i][1] for i in range(len(xray_train_ds))])
        X_train = []
        y_train = []

        for i in range(len(xray_train_ds)):
            result = xray_train_ds[i]
            if result is None:
                print(f"Skipping invalid sample {i} from client {client_idx}")
                continue
            image, label = result
            X_train.append(image.numpy())
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        n_train = y_train.shape[0]
        idxs = np.random.permutation(n_train)
        # net_dataidx_map = {client_idx: idxs}
        net_dataidx_map[client_idx] = idxs
        traindata_cls_counts = record_net_data_stats(y_train, idxs, n_classes)

        print(f"Client {client_idx} has data class: {traindata_cls_counts.shape}")
        print(f"Client {client_idx} has data class-num counts: {traindata_cls_counts}")

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        # all_net_dataidx_map.append(net_dataidx_map)
        all_traindata_cls_counts.append(traindata_cls_counts)

    print(all_traindata_cls_counts)
    all_traindata_cls_counts = np.array(all_traindata_cls_counts)

    return all_X_train, all_y_train, X_test, y_test, net_dataidx_map, all_traindata_cls_counts


def load_DatasetTIF_data(datadir, n_parties, dataset, train_distribution, val_distribution):
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if dataset == 'RS-5':
        n_classes = 5
    elif dataset == "RS-15":
        n_classes = 15
    all_X_train = []
    all_y_train = []
    net_dataidx_map = {}
    all_traindata_cls_counts = []
    # transform = transforms.Compose([transforms.ToTensor()])
    if train_distribution == 'noniid-1':
        train_dir = f"{datadir}/train/"
    elif train_distribution == 'noniid-2':
        train_dir = f"{datadir}/train_balanced/"
    elif train_distribution == 'centralized':
        train_dir = f"{datadir}/train_merged/"
    # load test set
    test_dir = f"{datadir}/val/"
    if val_distribution == 2:
        test_dir = f"{datadir}/val2/"
    xray_test_ds = ImageFolder_SingleDatasetTIF(test_dir, client_idx=None, transform=transform_test)

    X_test = np.array([xray_test_ds[i][0].numpy() for i in range(len(xray_test_ds))])
    y_test = np.array([xray_test_ds[i][1] for i in range(len(xray_test_ds))])

    for client_idx in range(n_parties):
        # load train set
        xray_train_ds = ImageFolder_SingleDatasetTIF(train_dir, client_idx=client_idx, transform=transform_train)

        # get features and labels
        # X_train = np.array([xray_train_ds[i][0].numpy() for i in range(len(xray_train_ds))])
        # y_train = np.array([xray_train_ds[i][1] for i in range(len(xray_train_ds))])
        X_train = []
        y_train = []

        for i in range(len(xray_train_ds)):
            result = xray_train_ds[i]
            if result is None:
                print(f"Skipping invalid sample {i} from client {client_idx}")
                continue
            image, label = result
            X_train.append(image.numpy())
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        n_train = y_train.shape[0]
        idxs = np.random.permutation(n_train)
        # net_dataidx_map = {client_idx: idxs}
        net_dataidx_map[client_idx] = idxs
        traindata_cls_counts = record_net_data_stats(y_train, idxs, n_classes)

        print(f"Client {client_idx} has data class: {traindata_cls_counts.shape}")
        print(f"Client {client_idx} has data class-num counts: {traindata_cls_counts}")

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        # all_net_dataidx_map.append(net_dataidx_map)
        all_traindata_cls_counts.append(traindata_cls_counts)

    print(all_traindata_cls_counts)
    all_traindata_cls_counts = np.array(all_traindata_cls_counts)
    # all_X_train = np.array(all_X_train)
    # all_y_train = np.array(all_y_train)
    return all_X_train, all_y_train, X_test, y_test, net_dataidx_map, all_traindata_cls_counts


def record_net_data_stats(y_train, dataidx, num_classes):
    net_cls_counts_dict = {}
    net_cls_counts_npy = np.array([])
    # num_classes = int(y_train.max()) + 1

    # for net_i, dataidx in net_dataidx_map.items():
    unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
    tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    # net_cls_counts_dict[net_i] = tmp
    tmp_npy = np.zeros(num_classes)
    for i in range(len(unq)):
        tmp_npy[unq[i]] = unq_cnt[i]
    net_cls_counts_npy = np.concatenate(
        (net_cls_counts_npy, tmp_npy), axis=0)
    net_cls_counts_npy = np.reshape(net_cls_counts_npy, (-1, num_classes))

    data_list = []
    # for net_id, data in net_cls_counts_dict.items():
    #     n_total=0
    #     for class_id, n_data in data.items():
    #         n_total += n_data
    #     data_list.append(n_total)
    print('Data distribution over clients (each row for a client): ')
    print(net_cls_counts_npy.astype(int), '\n')
    return net_cls_counts_npy


def partition_data(args):
    dataset, datadir, partition, n_parties, beta, n_niid_parties, global_imbalanced, download, val_distribution, dataset_form = \
        args.dataset, args.datadir, args.partition, args.n_parties, args.beta, args.n_niid_parties, \
            args.train_global_imb, args.download_data, args.val_distribution, args.dataset_form

    if 'RS' in dataset:
        if dataset_form == "imagefolder":
            _, y_train, _, _, net_dataidx_map, traindata_cls_counts = load_DatasetTIF_data(datadir, n_parties=n_parties,
                                                                                           dataset=dataset,
                                                                                           train_distribution=partition,
                                                                                           val_distribution=val_distribution)
        elif dataset_form == "parquet":
            _, y_train, _, _, net_dataidx_map, traindata_cls_counts = load_parquet_data(datadir, n_parties=n_parties,
                                                                                        dataset=dataset,
                                                                                        train_distribution=partition,
                                                                                        val_distribution=val_distribution)
    return (net_dataidx_map, traindata_cls_counts)


def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(mdl.state_dict())
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).cuda())
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    if len(target) == 1:
                        out = out.unsqueeze(0)
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
    if was_training:
        model.train()
    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss
    return correct / float(total), avg_loss


def get_train_dataloader(args, client_idx, test_bs=32):
    dataset, datadir, train_bs, model, download, dataset_form = (
        args.dataset, args.datadir, args.batch_size,
        args.model, args.download_data, args.dataset_form
    )
    size = (64, 64)
    if model == "vit-b-16":
        size = (224, 224)

    if 'RS' in dataset:
        if dataset_form == "imagefolder":
            dl_obj = ImageFolder_SingleDatasetTIF
            data_dir = datadir
        elif dataset_form == "parquet":
            dl_obj = ParquetDataset
            data_dir = os.path.join(datadir, "train.parquet")
        else:
            raise ValueError("Unsupported dataset format. Use 'imagefolder' or 'parquet'.")

        transform_train = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if args.partition in ['noniid-1', 'noniid-2', 'centralized']:
            if dataset_form == "parquet":
                map_root = os.path.join(datadir, 'map', 'map-RS5-NIID1.json')
            else:
                map_root = None

            if dataset_form == "parquet":
                train_ds = dl_obj(
                    parquet_path=data_dir,
                    map_path=map_root,
                    client_idx=client_idx if args.partition != 'centralized' else None,
                    transform=transform_train
                )
            else:
                train_dir = 'train_merged' if args.partition == 'centralized' else f'train'
                train_client_path = os.path.join(datadir, train_dir)
                train_ds = dl_obj(
                    root=train_client_path,
                    client_idx=client_idx if args.partition != 'centralized' else None,
                    transform=transform_train
                )

        train_dl = data.DataLoader(
            dataset=train_ds,
            batch_size=train_bs,
            drop_last=True,
            shuffle=True,
        )

    return train_dl


def get_test_dataloader(args, test_bs=32):
    dataset, datadir, model, dataset_form = (
        args.dataset, args.datadir, args.model, args.dataset_form
    )
    size = (64, 64)
    if model == "vit-b-16":
        size = (224, 224)

    if 'RS' in dataset:
        if dataset_form == "imagefolder":
            dl_obj = ImageFolder_SingleDatasetTIF
            test_dir = 'val2/' if args.val_distribution == 2 else 'val/'
            test_client_path = os.path.join(datadir, test_dir)
        elif dataset_form == "parquet":
            dl_obj = ParquetDataset
            test_parquet = os.path.join(datadir, "test.parquet")
            test_client_path = test_parquet
            if dataset == 'RS-5':
                map_path = os.path.join(datadir, "map",
                                        "map-RS5-balanced.json" if args.val_distribution == 2 else "map-RS5-balanced.json")
            elif dataset == 'RS-15':
                map_path = os.path.join(datadir, "map",
                                        "map-balanced.json" if args.val_distribution == 2 else "map-balanced.json")
        else:
            raise ValueError("Unsupported dataset format. Use 'imagefolder' or 'parquet'.")

        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if dataset_form == "parquet":
            test_ds = dl_obj(
                parquet_path=test_client_path,
                map_path=map_path,
                client_idx=None,
                transform=transform_test
            )
        else:
            test_ds = dl_obj(
                root=test_client_path,
                client_idx=None,
                transform=transform_test
            )

        test_dl = data.DataLoader(
            dataset=test_ds,
            batch_size=test_bs,
            shuffle=False,
        )

    return test_dl


def gaussian_noise(data_shape, args, device):
    if args.dp_sigma is None:
        # delta_l = 2 * args.lr * args.dp_max_grad_norm / (args.sample_num / args.n_parties)
        delta_l = 2 * args.lr * args.dp_max_grad_norm / (args.sample_num)
        # sigma = np.sqrt(2 * np.log(1.25 / script_args.dp_delta)) / script_args.dp_epsilon
        q = args.sample_num / args.n_parties
        sigma = delta_l * math.sqrt(2 * q * args.comm_round * math.log(1 / args.dp_delta)) / args.dp_epsilon
    else:
        sigma = args.dp_sigma
    return torch.normal(0, sigma, data_shape).to(device)