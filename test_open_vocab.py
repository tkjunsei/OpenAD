import os
import argparse
from gorilla.config import Config
from os.path import join as opj
# utilsからset_random_seedやevaluation関数がインポートされている
from utils import *
import torch
from torch_cluster import fps

"""
学習したモデルの評価
テスト時コマンド例
$CUDA_VISIBLE_DEVICES=0 python3 test_open_vocab.py --config ./config/openad_pn2/full_shape_open_vocab_cfg.py --checkpoint <path to your checkpoint model> --gpu 0
"""

# Argument Parser
def parse_args():
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser(description="Test model on unseen affordances")
    # testするモデルのconfigファイル指定    
    parser.add_argument("--config", help="config file path")
    # testするモデルのチェックポイント指定(.t7?)
    parser.add_argument("--checkpoint", help="the dir to saved model")
    # gpu番号
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    args = parser.parse_args()
    return args

# main
if __name__ == "__main__":
    # parse設定
    args = parse_args()
    # config設定
    cfg = Config.fromfile(args.config)

    # logファイル設定(出力はconfigで指定されたwork_dir(OPENAD_PN2_ESTIMATION_Release)/result_{modelのタイプ}.log)
    logger = IOStream(opj(cfg.work_dir, 'result_' + cfg.model.type + '.log'))
    # シード値設定
    # シード値があるならば，ランダムにシード値を設定
    if cfg.get('seed', None) != None:
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
        
    # gpuが設定されているならば，設定上書き
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = 'cuda'   #cfg.training_cfg.gpu
    # 指定したconfigファイル，gpuを使ってモデルを構築
    model = build_model(cfg).cuda()
    # (エラーハンドリング)コマンドライン引数にcheckpointがないならば終了
    if args.checkpoint == None:
        print("Please specify the path to the saved model")
        exit()
    # コマンドライン引数でcheckpointを指定しているならば
    else:
        # 拡張子による条件分岐
        print("Loading model....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            model.load_state_dict(torch.load(args.checkpoint))
            print('done')
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])

    # configファイルからデータセット準備
    dataset_dict = build_dataset(cfg)       # build the dataset
    # データセットに基づいてデータローダーを構築
    loader_dict = build_loader(cfg, dataset_dict)       # build the loader
    # 検証用データローダーを構築．ただし，作成したデータローダーにval_loaderが存在しないならNoneを返す
    val_loader = loader_dict.get("val_loader", None)
    # 検証用Affordanceを指定
    val_affordance = cfg.training_cfg.val_affordance
    # 検証データからモデルを評価
    mIoU = evaluation(logger, cfg, model, val_loader, val_affordance)