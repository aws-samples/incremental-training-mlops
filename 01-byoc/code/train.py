import os
import time
import copy
import datetime
from pprint import pformat
import logging
import torch
import numpy as np
from dataset import SoundDataset
from dataloader import SoundDataLoader
from config import ParameterSetting
from workflow import * 
import pkbar

logger = logging.getLogger(__file__)


def main():
    parser = ArgumentParser()
    # data or model path setting
    parser.add_argument("--csv_path", type=str, default='/DATA/hucheng/competition/official/preliminary/after_trim/meta_train.csv', help='the path of train csv file')
    parser.add_argument("--data_dir", type=str, default="/DATA/hucheng/competition/official/preliminary/after_trim/train", help="the directory of sound data")
    parser.add_argument("--save_root", type=str, default="./results", help="the root of results")
    parser.add_argument("--model_file", type=str, default="./results/final_mode.pkl", help="the root of results")
    parser.add_argument("--resume", type=str, default=None, help="the path of resume training model")
    # training parameter setting
    parser.add_argument("--model_name", type=str, default='VGGish', choices=['VGGish'], help='the algorithm we used')
    parser.add_argument("--val_split", type=float, default=0.1, help="the ratio of validation set. 0 means there's no validation dataset")
    parser.add_argument("--epochs", type=int, default=20, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam"])
    parser.add_argument("--scheduler", type=str, default="steplr", choices=["steplr"])
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--num_class", type=int, default=6, help="number of classes")
    parser.add_argument("--normalize", type=str, default=None, choices=[None, 'rms', 'peak'], help="normalize the input before fed into model")
    parser.add_argument("--preload", action='store_true', default=False, help="whether to convert to melspectrogram first before start training")
    # data augmentation setting
    parser.add_argument("--spec_aug", action='store_true', default=False)
    parser.add_argument("--time_drop_width", type=int, default=64)
    parser.add_argument("--time_stripes_num", type=int, default=2)
    parser.add_argument("--freq_drop_width", type=int, default=8)
    parser.add_argument("--freq_stripes_num", type=int, default=2)
    # proprocessing setting
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--nfft", type=int, default=200)
    parser.add_argument("--hop", type=int, default=80)
    parser.add_argument("--mel", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    ##################
    # config setting #
    ##################

    params = ParameterSetting(args.csv_path, args.data_dir, args.save_root, args.model_file, args.model_name, args.val_split,
                              args.epochs, args.batch_size, args.lr, args.num_class,
                              args.time_drop_width, args.time_stripes_num, args.freq_drop_width, args.freq_stripes_num,
                              args.sr, args.nfft, args.hop, args.mel, args.resume, args.normalize, args.preload,
                              args.spec_aug, args.optimizer, args.scheduler)

    if not os.path.exists(params.save_root):
        os.mkdir(params.save_root)
        print("create folder: {}".format(params.save_root))
        if not os.path.exists(os.path.join(params.save_root, 'snapshots')):
            os.mkdir(os.path.join(params.save_root, 'snapshots'))
        if not os.path.exists(os.path.join(params.save_root, 'log')):
            os.mkdir(os.path.join(params.save_root, 'log'))

    ###################
    # model preparing #
    ###################

    model = prepare_model(params)

    ##################
    # data preparing #
    ##################

    print("Preparing training/validation data...")
    dataset = SoundDataset(params)

    train_dataloader = SoundDataLoader(dataset, batch_size=params.batch_size, shuffle=True, validation_split=params.val_split, pin_memory=True)
    val_dataloader = train_dataloader.split_validation()

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataloader.sampler), 'val': len(train_dataloader.valid_sampler)}
    print("train size: {}, val size: {}".format(dataset_sizes['train'], dataset_sizes['val']))

    ##################
    # model training #
    ##################

    # start to train the model
    train_model(model, params, dataloaders, dataset_sizes)

if __name__ == '__main__':
    main()
