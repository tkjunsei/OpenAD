import torch

file_paths = [
    '/home/junsei/Downloads/GitHub/OpenAD/teacher_model/poinet++/best_model_1.pth',
    '/home/junsei/Downloads/GitHub/OpenAD/teacher_model/poinet++/best_model_cls_6.pth',
    '/home/junsei/Downloads/GitHub/OpenAD/teacher_model/poinet++/best_model_cls.pth',
    '/home/junsei/Downloads/GitHub/OpenAD/teacher_model/poinet++/best_model.pth',    
    '/home/junsei/Downloads/GitHub/OpenAD/teacher_model/poinet++/point_techer_pretrained.pth',
    torch.load('/home/junsei/Downloads/GitHub/OpenAD/teacher_model/poinet++/model.cls.2048.t7'),
    torch.load('/home/junsei/Downloads/GitHub/OpenAD/teacher_model/poinet++/model.partseg.t7')
]

results = []

for file_path in file_paths:
    data = torch.load(file_path)
    epoch = data.get('epoch', 'Unknown')
    class_avg_iou = data.get('class_avg_iou', None)
    instance_acc = data.get('instance_acc', None)
    class_acc = data.get('class_acc', None)

    results.append({
        'epoch': epoch,
        'class_avg_iou': class_avg_iou,
        'instance_acc': instance_acc,
        'class_acc': class_acc
    })

# 出力
for result in results:
    print(result)
