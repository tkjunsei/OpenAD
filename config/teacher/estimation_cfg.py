import os
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

"""
生徒モデルの学習の際に使用されるconfigファイルの定義
"""

exp_name = "teacher_ESTIMATION_Release_1123"
# work_dir == ./log/openad_pn2/{exp_name}
work_dir = opj("./log/openad_pn2", exp_name)
seed = 1
# ディレクトリがなければ作成
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

# 学習率スケジューラの設定
# 用いるスケジューラはLambdaLR，初期学習率0.001,20ステップごとに減衰率0.5,最小学習率は1e-5
scheduler = dict(
    type='lr_lambda',
    lr_lambda=PN2_Scheduler(init_lr=0.001, step=20,
                            decay_rate=0.5, min_lr=1e-5)
)

# 最適化アルゴリズムの設定
# Adam_optimizerを使用．初期学習率0.001,モーメント項の係数(0.9,0.999),L2正則化を行う
optimizer = dict(
    type='adam',
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4
)

# 使用するモデルの設定
# 使用するモデルの名前:openad_pn2, モデルの初期重みの指定
model = dict(
    type='teacher_point2',
    weights_init='pn2_init'
)

# トレーニングの設定
# estimate:推定タスクを有効化
# partial:
# rotate:回転変換(z軸回転や3次元回転も可能,前のopenADでは実験されていた)
# semi:
# rotate_type:
# batch_size:ミニバッチサイズ
# epoch:トレーニングエポック数
# seed:
# dropout:ドロップアウト率
# gpu:使用するGPU番号
# workflow:トレーニングと検証を1エポックずつ交互に実行
# bn_momentum:BatchNormモメンタムの設定,初期値0.1,減衰率0.1,20エポックごとに調整
# _affordance:トレーニング,検証,ゼロベースラベルのaffordanceカテゴリ
training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate='None',  # z,so3
    semi=False,
    rotate_type=None,
    batch_size=128,
    epoch=200,
    seed=1,
    dropout=0.5,
    gpu='4',
    workflow=dict(
        train=1,
        val=1
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),
    train_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
    val_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
)

# データセットの設定
# data_root:データセットのパス
# Affordanceカテゴリ
data = dict(
    data_root = '/home/junsei/Downloads/GitHub/OpenAD/drive',
    category = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none']
)
