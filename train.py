# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from os.path import join as opj
import torch
import torch.nn as nn
import numpy as np
from gorilla.config import Config
import models
import loss
from utils import *
import argparse
# from models.openad_dgcnn import MultiHeadedSelfAttention
from models.openad_pn2 import MultiHeadedSelfAttention

import wandb

"""
モデルの学習を行う
実行コマンド例
$CUDA_VISIBLE_DEVICES=0 python3 train.py --config ./config/openad_pn2/full_shape_cfg.py --config_teacher './config/teacher/estimation_cfg.py' --checkpoint_teacher <path to your checkpoint teacher model> --work_dir ./log/openad_pn2/OPENAD_PN2_FULL_SHAPE_Release/ --gpu 0
--config
.py形式のconfigファイル(./config/openad_pn2/estimation_cfg.py等，openvocab_cfg.pyはテスト用)
--config_teacher
./drive内にある.pthもしくは.t7形式のファイル
--work_dir
保存先ファイル名（ちゃんと調べられてないが，{work_dir}_poinet_{数字}に保存される気がする
--gpu
使用するGPUの番号，githubでは一つのGPUで学習をすることで最高のパフォーマンスを得られると書いてある
"""

# Argument Parser
def parse_args():
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser(description="Train a model")
    # 学習させる生徒モデルのconfigファイルのパス
    parser.add_argument("--config", help="train config file path")
    # 教師モデルのconfigファイルのパス
    parser.add_argument("--config_teacher", help="train config file path")
    # ログと学習が完了した生徒モデルの保存場所
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    # 学習に使うGPUの番号
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    # 教師モデルのチェックポイントのパス(checkpoint == 学習済みモデルのパラメータファイル)
    parser.add_argument(
        "--checkpoint_teacher",
        type=str,
        default=None,
        help="The checkpoint to be resume"
    )
    # 生徒モデルのチェックポイントのパス(checkpoint == 学習済みモデルのパラメータファイル)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume training"
    )
    args = parser.parse_args()
    return args

# main
if __name__ == "__main__":
    # パースのオブジェクト作成
    args = parse_args()
    # 生徒モデル用のconfigファイルの読み込み
    cfg = Config.fromfile(args.config)
    # すでに保存ディレクトリがconfigファイルに書き込まれていれば，work_dirをコマンドライン引数で上書き
    if args.work_dir != None:
        cfg.work_dir = args.work_dir
    # すでに使用gpu番号がconfigファイルに書き込まれていれば，gpuをコマンドライン引数で上書き
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
    # 教師モデル用のconfigファイルを読み込む
    cfg_teacher  = Config.fromfile(args.config_teacher)

    # WandBの初期化
    wandb.init(
        project="OpenAD-training_20241123",
        config=cfg.training_cfg,
        name=cfg.work_dir.split("/")[-1]
    )


    # logファイルの作成
    # opj = os.path.joinの省略形
    # {work_dir}/run.logにlogファイルを作成
    logger = IOStream(opj(cfg.work_dir, 'run.log'))     # opj is os.path.join
    # 使用するGPUを環境変数CUDA_VISIBLE_DEVICESに保存
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    # 利用するGPUの数を計算
    num_gpu = len(cfg.training_cfg.gpu.split(','))      # number of GPUs to use
    # logに出力
    logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))      # write to the log file
    # シード値を上書き
    if cfg.get('seed') != None:     # set random seed
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
    # configから生徒モデルを構築
    model = build_model(cfg,is_teacher=False).cuda()     # build the model from configuration
    # configから教師モデルを構築
    teacher_model = build_model(cfg_teacher, is_teacher=True).cuda()

    # kdmodel：知識蒸留(Knowlede Distillation)を行うモデル，引数の意味は不明
    # kd_models =  MultiHeadedSelfAttention(515,64,12,0.1)#dgcnn
    kd_models =  MultiHeadedSelfAttention(128,64,12,0.1)#poinet
    kd_models = kd_models.cuda()
    # if train on multiple GPUs, not recommended for PointNet++(複数のGPUで学習をお粉ならPointNet++で行うのはおすすめしないよ)
    # 使うGPUが複数あるならデータ並列化を行う
    if num_gpu > 1:
        model = nn.DataParallel(model)
        logger.cprint('Use DataParallel!')

    # if the checkpoint file is specified, then load it(checkpointモデルを使うなら指定されたものをロードする)
    # 生徒モデルのcheckpointが存在するなら
    if args.checkpoint != None:
        print("Loading checkpoint....")
        # 拡張子のみの読み込み
        _, exten = os.path.splitext(args.checkpoint)
        # 教師モデルの読み込み
        # .t7なら
        if exten == '.t7':
            teacher_model = nn.DataParallel(teacher_model)
            teacher_model.load_state_dict(torch.load(args.checkpoint_teacher))
            # model = nn.DataParallel(model)
            # model.load_state_dict(torch.load('/home/tuan.vo1/IROS2023_Affordance-master/log/openad_dgcnn/OPENAD_PN2_ESTIMATION_Release_globallocal/best_model.t7'))
        # .pthなら
        elif exten == '.pth':
            check = torch.load(args.checkpoint_teacher)
            teacher_model.load_state_dict(check['model_state_dict'])
            # check_model = torch.load('/home/tuan.vo1/IROS2023_Affordance-master/log/openad_dgcnn/OPENAD_PN2_ESTIMATION_Release_globallocal/best_model.t7')
            # model.load_state_dict(check_model['model_state_dict'])
    # else train from scratch
    # 生徒モデルのチェックポイントがないなら0から学習する(教師モデルについての記載は割愛されている)
    else:
        print("Training from scratch!")

    # configファイルからデータセットとそれに対応するデータローダ，損失関数，最適化アルゴリズムの構築
    dataset_dict = build_dataset(cfg)       # build the dataset
    loader_dict = build_loader(cfg, dataset_dict)       # build the loader
    train_loss = build_loss(cfg)        # build the loss function
    optim_dict = build_optimizer(cfg, model, kd_models)        # build the optimizer
    # construct the training process(学習の条件を辞書として格納)
    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        loss=train_loss,
        optim_dict=optim_dict,
        logger=logger,
        teacher_model = teacher_model,
        kd_models = kd_models
    )

    # 学習の実行(Trainerクラスはutils.にある)
    task_trainer = Trainer(cfg, training)
    task_trainer.run()



