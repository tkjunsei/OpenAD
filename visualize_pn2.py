import os
import torch
import torch.nn as nn
from utils import *
import clip
from models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
import pickle
import torch.nn.functional as F
import numpy as np
import open3d as o3d	# Main library that we use for visualization, refer to its docs for details
from scipy.spatial.transform import Rotation as R	# For rotation matrices


"""
Re-define the point cloud network PointNet++. You will need to modify this according to the point cloud network you defined
自分が作成したnetworkにこのコードを修正してね
"""
class PN2(nn.Module):
    """点群データから特徴点を抽出"""
    # normal_channel:点群に法線情報を含めるかどうか(defaultはFalse)
    def __init__(self, normal_channel=False):
        # 初期化
        super(PN2, self).__init__()

        # 点群への法線情報
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        # アップサンプリング：データセット内のクラスの不均衡に対処する，データ処理及び最適化の手法

        # sa(Set Abstraction)
        # mlp:多層パーセプトロン？
        # 点群の階層的な特徴抽出（ココらへんの話は論文PointNet++参照：https://arxiv.org/pdf/1706.02413）
        # 複数のスケール(半径)で点群をグループ化し特徴を抽出
        # sa1:最初の抽象化レイヤー（512点，3つのスケール0.1, 0.2, 0.4）
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # sa2:次の抽象化レイヤー（128点，2つのスケール0.4, 0.8）
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # 全点群の特徴を統合
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # fp(Feature Propagation)
        # 抽出された特徴を高解像度にアップサンプリング（抽象化された特徴を統合し，細かい特徴を復元する）
        # check_model_weights.pyでチェックポイントの中身を確認する
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        # self.fp3 = PointNetFeaturePropagation(in_channel=1024, mlp=[128, 128])
        # self.fp2 = PointNetFeaturePropagation(in_channel=448, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=134+additional_channel, mlp=[128, 128])
        # １次元畳み込み層
        self.conv1 = nn.Conv1d(128, 512, 1)
        # バッチ正規化
        self.bn1 = nn.BatchNorm1d(512)
        # dropout設定
        self.drop1 = nn.Dropout(0.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    # 順伝播(点群データを入力し，抽象化された特徴を生成)
    def forward(self, xyz):
        """順伝播"""
        # Set Abstraction layers
        xyz = xyz.contiguous()
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_xyz = xyz
            l0_points = xyz

        # sa1, sa2, sa3：点群の抽象化
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size())
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size())
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)
        # print(l0_points.size())
        x = self.bn1(self.conv1(l0_points))
        return x


# Define the rotation matrices. A rotation matrix is needed when you want to see other angles of the object for better visualization
# Feel free to declare other rotation matrices
rotation_matrices = [np.identity(3, dtype='float32'),	# Default: Identity matrix
                    np.array(R.from_euler('y', 90, degrees=True).as_matrix(), dtype='float32'),
                    np.array(R.from_euler('xy', [-20, 40], degrees=True).as_matrix(), dtype='float32'),
                    np.array(R.from_euler('x', -20, degrees=True).as_matrix(), dtype='float32'),
                    np.array(R.from_euler('x', -45, degrees=True).as_matrix(), dtype='float32'),
                    np.array(R.from_euler('y', 20, degrees=True).as_matrix(), dtype='float32'),
                    np.array(R.from_euler('x', -50, degrees=True).as_matrix(), dtype='float32')]


# Define the list of colors for visualization. Each color is for each label
color_list = [(0., 0., 0.), (0., 0., 255.), (255., 0., 0.), (0., 255., 0.), (180., 125., 100.), (110., 65., 130.),
               (0., 255., 255.), (0., 0., 115.), (170., 50., 115.), (250., 115., 0.), (235., 255., 0.), (145., 105., 0.),
               (150., 150., 120.), (205., 210., 145.), (100., 190., 255.), (45., 45., 90.), (120., 45., 45.), (100., 0., 0.), (0., 50., 100.)]


# Define the main visualization process
if __name__ == "__main__":
    # Initialize the point cloud network
    pn2 = PN2().to('cuda')

    # 学習させたモデルのパス(best_model.t7)
    checkpoint = './log/openad_pn2/OPENAD_PN2_FULL_SHAPE_Release_11231539/best_model.t7'
    print("Loading checkpoint....")
    _, exten = os.path.splitext(checkpoint)
    
    # Load the trained point cloud network's weights
    if exten == '.t7':
        checkpoint_dict = torch.load(checkpoint)
        pn2_dict = pn2.state_dict()
        # pn2_dictとcheckpoint_dictの一致するパラメータのみを更新
        checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if (k in pn2_dict and v.size() == pn2_dict[k].size())}
        pn2_dict.update(checkpoint_dict)
        pn2.load_state_dict(pn2_dict)
    
    # Load the CLIP text encoder
    clip_model, _ = clip.load("ViT-B/32", device='cuda')

    # Load point cloud data
    # 点群データのパス(full_shape_val.pkl)
    with open('./drive/full_shape_val_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Define the list of object names
    object_names = ('Bag', 'Bed', 'Bowl', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone', 'Faucet', 'Hat', 'StorageFurniture', 'Keyboard'
   , 'Knife', 'Laptop', 'Microwave', 'Mug', 'Refrigerator', 'Chair', 'Scissors', 'Table', 'TrashCan', 'Vase', 'Bottle')
    
    """
    Define the ```all_objects``` dictionary whose keys are object names and values are the corresponding lists of objects
    """
    all_objects = {}
    for obj in object_names:
        exec("all_objects['" + obj + "'] = []")               
    for obj in data:
        all_objects[obj["semantic class"]].append(obj)

    # open3Dによる点群データを可視化するための設定
    # Configure the important elements for visualization. Refer to Open3D's docs for more information
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 10.0 # Size of points in the point cloud. Feel free to change it according to your preference


    # Main loop
    while True:
    	# Input the affordance texts. The input can be arbitrary, e.g., <none,cut,stab> or <none, grasp, support> or <none,sittable>. Just need to make sure that the 'none' label comes first,
    	# so that it can be visualized in the black color (check the ```color_list```)
        affordance_texts = [item for item in input("Enter the texts : ").split(',')]
        
        # Pick the suitable rotation matrix
        matrix_index = int(input("Enter the index of rotation matrix: "))

        
    	# Input the object type that you want to visualize
        obj = input("Choose an object type: ")	# For example, Bag or Bed or Mug
        
        # リスト(object_names)に存在しないカテゴリ
        if obj not in object_names:	# If the entered object is not in the list of object names
            print("Type not existed!")
            continue
	
	# Loop through the corresponding list of objects, visualize one object at a time
        for id in range(len(all_objects[obj])):
            if id % 10 == 0: # Pause every 10 objects to ask for continuation
                message = input("Do you want to continue? ")
                if message == 'n':	# If want to pause	
                    break
            
            # numpy点群データをPyTorchテンソルに変換
            # Get the input point cloudpip install numpy==1.23.5
            xyz = torch.from_numpy(all_objects[obj][id]['full_shape']['coordinate']).unsqueeze(0).permute(0, 2, 1).cuda()
            with torch.no_grad():
                pn2.eval()
                clip_model.eval()
                
                # Compute the per-point features(各点の特徴量を計算)
                point_features = pn2(xyz).float().permute(0, 2, 1) # [1, 2048, 512]
                
                # 入力テキスト特徴量の計算
                """
                Compute the text features of the entered affordance
                """
                tokens = clip.tokenize(affordance_texts).to('cuda')
                text_features = clip_model.encode_text(tokens).to('cuda').float().permute(1, 0) # [512, 19]
                
                # 点特徴とテキスト特徴の相関計算(点群とaffordance_textsの相関計算)
                """
                Correlate the text and point features, and compute the final labels
                """
                result = F.softmax(pn2.logit_scale * ((point_features @ text_features) / (torch.norm(point_features, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True))).permute(0, 2, 1), dim=1).cpu().numpy()
                result = np.squeeze(result)
                result = np.argmax(result, axis=0)


            # 点群データの色付けと可視化
            cloud = o3d.geometry.PointCloud()	# Initialize a point cloud instance
            cloud.points = o3d.utility.Vector3dVector(all_objects[obj][id]['full_shape']['coordinate'] @ rotation_matrices[matrix_index])	# Declare the point coordinate (after rotation)
            
            """
            Declare the color for each point in the point cloud
            """
            color =  np.concatenate([np.array(color_list[i]).reshape((1, -1)) for i in result], axis=0) / 255.
            cloud.colors = o3d.utility.Vector3dVector(color)
            
            # save the result
            # 点群データを保存
            output_ply_path = f"pointcloud_{obj}_{id}.ply"  # オブジェクト名とIDを含むファイル名
            o3d.io.write_point_cloud(output_ply_path, cloud)
            print(f"Saved point cloud to {output_ply_path}")


            # Visualize the result
            # o3d.visualization.draw([{'name': 'Visualization', 'geometry': cloud, 'material': mat}], title='VISUALIZATION' + '_' + obj, show_skybox=False)
