# Copyright (c) Gorilla-Lab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join as opj
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from utils.eval import evaluation
from time import time
import torch.optim as optim
from torch.autograd import Variable


import torch as T

import wandb
import pynvml

# -----------------------------------------------------------

class ContrastiveLoss(T.nn.Module):
  """対照学習(似たものを似たベクトルにする学習)における損失関数"""
  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
    self.m = m  # margin or radius

  def forward(self, y1, y2, d=0):
    # d = 0 means y1 and y2 are supposed to be same
    # d = 1 means y1 and y2 are supposed to be different
    # d = 0なら類似サンプル，1なら相違サンプル
    
    # ユークリッド距離(Euclidean distance)の計算
    euc_dist = T.nn.functional.pairwise_distance(y1, y2)

    if d == 0:
      # ユークリッド距離の二乗を計算
      return T.mean(T.pow(euc_dist, 2))  # distance squared
    else:  # d == 1
      delta = self.m - euc_dist  # sort of reverse distance
      delta = T.clamp(delta, min=0.0, max=None)
      return T.mean(T.pow(delta, 2))

class Trainer(object):
    """モデルの学習，検証，テスト，実行を行う"""
    # Trainerクラスの初期化，各変数の定義・初期化
    def __init__(self, cfg, running):
        # configファイルの内容を取得
        super().__init__()
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        # ./log/openad_pn2/OPENAD_PN2_FULL_SHAPE_Release_.../events.out.tfevents.~~~~.node1を作成する
        self.writer = SummaryWriter(self.work_dir)
        # 各設定をrunnning辞書に格納
        self.logger = running['logger']
        self.model = running["model"]
        self.teacher_model = running["teacher_model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        # lader_dictの中の学習，検証，ラベル等についての情報を取得(データローダー)
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.val_loader = self.loader_dict.get("val_loader", None)
        self.train_unlabel_loader = self.loader_dict.get(
            "train_unlabel_loader", None)
        if self.train_unlabel_loader is not None:       # for semi task only, not care for now
            self.unlabel_loader_iter = iter(self.train_unlabel_loader)
        self.test_loader = self.loader_dict.get("test_loader", None)
        self.loss = running["loss"]
        self.optim_dict = running["optim_dict"]
        self.optimizer = self.optim_dict.get("optimizer", None)
        self.scheduler = self.optim_dict.get("scheduler", None)
        self.epoch = 0
        self.best_val_mIoU = 0.0
        self.best_val_mIoU_zeroshot = 0.0
        self.bn_momentum = self.cfg.training_cfg.get('bn_momentum', None)
        self.train_affordance = cfg.training_cfg.train_affordance
        self.val_affordance = cfg.training_cfg.val_affordance
        self.zero_shot_Affordance = cfg.training_cfg.zero_affordance
        self.kd_models = running['kd_models']
        # optimiserの初期設定
        self.kd_optimizer  = optim.Adam([{'params':self.kd_models.parameters()}], lr=0.0005)
        # 知識蒸留
        self.kd_loss = torch.nn.MSELoss(reduction='mean')
        # 対照学習の損失関数
        self.contrastiveloss = ContrastiveLoss()
        
        return

    # 学習
    def train(self):
        """学習時のパラメータ設定"""
        train_loss = 0.0
        count = 0.0
        # .train()はここで定義されているtrain()メソッドではなく，PyTorchのtorch.nn.Moduleクラスで提供されるメソッド
        # training mode : BatchNorm層でミニバッチごとの平均・分散を計算する．Dropout層で過学習防止の為にランダムにニューロンを無効化する
        # evaluation mode : BatchNorm層で学習済みの統計量(平均・分散)を使用する(変化しない).Dropout層で全ニューロンンを有効化する

        # modelをトレーニングモードに変更
        self.model.train()
        # kd_modelsをトレーニングモードに変更
        self.kd_models.train()
        # teacher_modelを評価モードに変更
        self.teacher_model.eval()
        # 教師モデルのパラメータを固定(教師モデルが学習しないように設定)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        num_batches = len(self.train_loader)
        start = time()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
        # batchごとの処理
        for data, data1, label, _, cats in tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9):
            # ラベルなしデータが設定されていたら
            if self.train_unlabel_loader is not None:
                # データローダから次のバッチを取得
                try:
                    ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                    ul_data, ul_data1 = ul_data.float(), ul_data1.float()
                # データが尽きた場合の例外処理
                except StopIteration:
                    self.unlabel_loader_iter = iter(self.train_unlabel_loader)
                    ul_data, ul_data1, _, _ = next(self.unlabel_loader_iter)
                    ul_data, ul_data1 = ul_data.float(), ul_data1.float()

            # dataとlabelをgpuに転送
            data, label = data.float().cuda(), label.float().cuda()
            
            # 入力データの次元の順番を変更
            data = data.permute(0, 2, 1)
            # 不要な次元を削除し，ラベルを整数型に変換，メモリ配置を連続にする
            label = torch.squeeze(label).long().contiguous()
            batch_size = data.size()[0]
            num_point = data.size()[2]
            # print('check input:',data.shape)
            self.optimizer.zero_grad()
            if self.train_unlabel_loader is not None:
                ul_data = ul_data.cuda().float().permute(0, 2, 1)
                data_ = torch.cat((data, ul_data), dim=0)  # VAT
                afford_pred = self.model(data_, self.train_affordance)
            else:
                data_st = data[:, :3, :]
                # afford_pred(ラベル予測), 特徴量pc_relations_st(studentモデルの点群間の関係情報), local_in_global_st(studentモデルの局所的及びグローバルな特徴), pc_relation_tc(teacherモデルの点群間の関係情報), local_in_relation_tc(teacherモデルの局所的及びグローバルな特徴)
                afford_pred, pc_relations_st, local_in_global_st = self.model(data_st, self.train_affordance)
                _, pc_relations_tc, local_in_global_tc = self.teacher_model(data, self.train_affordance)
                # pc_relations_tc = Variable(pc_relations_tc, requires_grad=False)
                # local_in_global_tc = Variable(local_in_global_tc, requires_grad=False)

            afford_pred = afford_pred.contiguous()
            if self.train_unlabel_loader is not None:
                l_pred = afford_pred[:batch_size, :, :]  # VAT
                ul_pred = afford_pred[batch_size:, :, :]  # VAT
                loss = self.loss(self.model, data, ul_data,
                                 l_pred, label, ul_pred, self.epoch)  # VAT
            else:
                # 損失関数
                # メインタスクの損失
                loss = self.loss(afford_pred, label)
                # print('vo day')
                # loss_struct_distill =  torch.square(pc_relations_st - pc_relations_tc).mean([2, 1, 0]) #self.kd_loss(pc_relations_st,pc_relations_tc)
                # loss_struct_distill =  self.kd_loss(pc_relations_st,pc_relations_tc)#mse
                # loss_struct_distill = (1 - torch.nn.CosineSimilarity()(pc_relations_st, pc_relations_tc)).mean() #cosine
                # 対照学習の損失（生徒モデルと教師モデルの特徴ベクトル間の類似度を最適化)
                loss_struct_distill = self.contrastiveloss(pc_relations_st, pc_relations_tc,0)
                
                # loss_struct_distill_global =   self.kd_loss(l0_points_org_st,l0_points_org_tc) #torch.square(l0_points_org_st - l0_points_org_tc).mean([2, 1, 0])
                # loss_total = loss+0.5*loss_struct_distill+loss_struct_distill_global
                # local_in_global_att = self.kd_models(local_in_global_st,local_in_global_tc) # for dgcnn
                # 知識蒸留の損失
                local_in_global_att = self.kd_models(local_in_global_tc) #for_poinet
                # print(local_in_global_att.shape)
                # glob_ftc = self.kd_models(l0_points_org_tc)
                # loss_kd =    torch.square(local_in_global_st-pc_relations_tc).mean([2, 1, 0]) #DGCNN#self.kd_loss(glob_fst,glob_ftc)
                # loss_kd =   torch.square(local_in_global_st-local_in_global_tc).mean([2, 1, 0]) #pointnet
                #loss_kd_gl =   self.kd_loss(local_in_global_st,local_in_global_tc)
                # local_in_global_att = local_in_global_att.permute(0,2,1)
                # print(local_in_global_att.shape)
                # print(pc_relations_st.shape)
                loss_global_local = self.kd_loss(local_in_global_st,local_in_global_att) #mse
                # loss_global_local = (1 - torch.nn.CosineSimilarity()(local_in_global_st, local_in_global_att)).mean()#cosine
                loss_total = loss+loss_struct_distill+  loss_global_local  #+ loss_kd_gl 
                # loss_total = loss
            # 損失関数の勾配
            loss_total.backward()
            # loss_kd.backward(retain_graph=True)
            # 生徒モデルのパラメータを更新
            self.optimizer.step()
            # self.kd_optimizer.step()

            count += batch_size * num_point
            train_loss += loss.item()
        # 学習率スケジューラ
        # 学習率を更新
        self.scheduler.step()
        if self.bn_momentum != None:
            # epoch数に応じてBatchNormのモメンタム(移動平均の更新率)を変更
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch))
        # 結果をログに流す
        epoch_time = time() - start

        # WandBにエポックごとの損失と時間を記録
        wandb.log({
            "epoch":self.epoch,
            "train_loss":train_loss*1.0/num_batches,
            "train_time_sec":epoch_time/1
        })

        outstr = 'Train(%d), loss: %.6f, time: %d s' % (
            self.epoch, train_loss*1.0/num_batches, epoch_time//1)
        self.writer.add_scalar('Loss', train_loss*1.0/num_batches, self.epoch)
        self.logger.cprint(outstr)
        self.epoch += 1

    # 検証
    def val(self):
        """学習中に検証を行い，最良のモデル(best_model.t7)と最新のモデル(current_model.t7)を随時保存，更新"""
        self.logger.cprint('Epoch(%d) begin validating......' % (self.epoch-1))
        # mIoUの計算
        mIoU = evaluation(self.logger, self.cfg, self.model,
                         self.val_loader, self.val_affordance)

        # トレーニングに存在しないAffordanceに対しての性能評価
        mIoU_Zeroshot = evaluation(self.logger, self.cfg, self.model,
                         self.val_loader, self.zero_shot_Affordance)
        
        # W&Bに検証結果を記録
        wandb.log({
            "epoch": self.epoch,
            "val_avg_class_IoU": mIoU,
            "val_avg_class_IoU_zeroshot": mIoU_Zeroshot
        })


        # mIoUとmIoU_Zeroshotがベスト値を更新した場合のみモデルを保存する(保存先：best_model.t7)
        if mIoU >= self.best_val_mIoU and mIoU_Zeroshot >= self.best_val_mIoU_zeroshot:
            self.best_val_mIoU = mIoU
            self.best_val_mIoU_zeroshot = mIoU_Zeroshot
            self.logger.cprint('Saving model......')
            self.logger.cprint('Best mIoU: %f' % self.best_val_mIoU)
            self.logger.cprint('Best mIoU_Zero-shot: %f' % self.best_val_mIoU_zeroshot)
            torch.save(self.model.state_dict(),
                   opj(self.work_dir, 'best_model.t7'))

        # トレーニングを途中から再開できるように最新の状態のモデルも保存(保存先：current_model.t7)
        torch.save(self.model.state_dict(),
                      opj(self.work_dir, 'current_model.t7'))

    # テスト
    def test(self):
        """テストを行う"""
        self.logger.cprint('Begin testing......')
        evaluation(self.logger, self.cfg, self.model,
                   self.test_loader, self.val_affordance)
        return

    def get_gpu_usage(self):
        """
        GPU使用状況を取得し、辞書形式で返す
        """
        try:
            # NVMLの初期化
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            gpu_info = {}
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_info[f"gpu_{i}_name"] = name
                gpu_info[f"gpu_{i}_memory_used_MB"] = memory_info.used // (1024 * 1024)
                gpu_info[f"gpu_{i}_memory_total_MB"] = memory_info.total // (1024 * 1024)
                gpu_info[f"gpu_{i}_memory_free_MB"] = memory_info.free // (1024 * 1024)
                gpu_info[f"gpu_{i}_utilization_percent"] = utilization.gpu
                gpu_info[f"gpu_{i}_temperature_C"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # NVMLの終了
            pynvml.nvmlShutdown()
            return gpu_info

        except pynvml.NVMLError as error:
            print(f"Error with NVML: {error}")
            return {}

    # 実行
    def run(self):
        # EPOCH == トレーニングを実行するエポック数
        EPOCH = self.cfg.training_cfg.epoch
        # workflow：トレーニングと検証の回数を決めている
        workflow = self.cfg.training_cfg.workflow
        # テスト用データローダが存在するならテストモードを実行
        if self.test_loader != None:
            epoch_runner = getattr(self, 'test')
            epoch_runner()
        # テスト用データローダが存在しないなら
        else:
            # configで決めたエポック数まで学習を継続する
            while self.epoch < EPOCH:
                for key, running_epoch in workflow.items():
                    epoch_runner = getattr(self, key)
                    for e in range(running_epoch):
                        epoch_runner()
                        # WandBにエポック進行状況を記録
                        wandb.log({"epoch_progress":self.epoch/EPOCH})
                        gpu_info = self.get_gpu_usage()
                        wandb.log(gpu_info)
