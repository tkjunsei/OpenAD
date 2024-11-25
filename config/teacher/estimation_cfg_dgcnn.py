import os
from os.path import join as opj

"""
教師モデルの学習の際に使用されるconfigファイルの定義
"""

exp_name = "OPENAD_DGCNN_ESTIMATION_Release"
# work_dir == ./log/openad_dgcnn/{exp_name}
work_dir = opj("./log/openad_dgcnn", exp_name)
seed = 1
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

scheduler = dict(
    type='cos',
    T_max=200,
    eta_min=1e-3
)
optimizer = dict(
    type='sgd',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

model = dict(
    type='dgcnn_cls_teacher',
    k=40,
    emb_dims=1024,
    dropout=0.5,
)

training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate='None',  # z,so3
    semi=False,
    rotate_type=None,
    batch_size=24,
    epoch=200,
    seed=1,
    dropout=0.5,
    gpu='4',
    workflow=dict(
        train=1,
        val=1
    ),
    train_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
    val_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
)

data = dict(
    data_root='/home/tuan.vo1/IROS2023_Affordance-master/Data',
    category=['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
              'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
              'listen', 'wear', 'press', 'cut', 'stab', 'none']
)