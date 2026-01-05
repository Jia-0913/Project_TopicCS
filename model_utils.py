import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image


# ==========================================
# 1. 修复后的模型类 (与训练代码保持完全一致)
# ==========================================
class WideResNetClassifier(nn.Module):
    def __init__(self, backbone, noise_std=0.1, dropout_rate=0.5):
        super().__init__()
        # 注意：训练代码中并没有使用 GaussianNoise，所以这里移除或保留不影响加载，
        # 但为了逻辑一致，推理(prediction)阶段不需要噪声，我们保持简洁。
        self.backbone = backbone

        # 自动检测输入特征数
        if hasattr(backbone, 'num_features'):
            in_features = backbone.num_features
        elif hasattr(backbone, 'fc'):
            in_features = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):
            in_features = backbone.classifier.in_features
        else:
            in_features = backbone.classifier.in_features

        # --- 这里是关键修复 ---
        # 必须匹配: Linear(512) -> BN -> ReLU -> Dropout -> Linear(128) -> ReLU -> Linear(2)
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),  # 修正：从 256 改为 512
            nn.BatchNorm1d(512),  # 修正：顺序调整，且维度为 512
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),  # 修正：输入从 256 改为 512
            nn.ReLU(),
            # 修正：训练代码中这里没有 Dropout，直接接输出层
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # 训练代码中 forward 只有这两步
        x = self.backbone(x)
        x = self.head(x)
        return x


# ==========================================
# 2. 加载逻辑
# ==========================================
MODEL_PATH = "cifake_model_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
# 增加 map_location 确保在只有 CPU 的机器上也能运行
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 读取配置
config = checkpoint.get('model_config', {'backbone_name': 'tf_efficientnet_b0', 'noise_std': 0.1, 'dropout_rate': 0.5})
data_config = checkpoint.get('data_config',
                             {'img_size': (224, 224), 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                              'class_names': ['Real', 'Fake']})

print(f"Model Info: {checkpoint.get('training_info', {})}")

# 初始化空模型结构
base_model = timm.create_model(config['backbone_name'], pretrained=False, num_classes=0)

# 初始化分类器
model = WideResNetClassifier(
    backbone=base_model,
    noise_std=config.get('noise_std', 0.1),
    dropout_rate=config.get('dropout_rate', 0.5)
)

# 加载权重 (此时应该不会再报错了)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()  # 切换到评估模式

print("✅ Model loaded successfully!")
print(f"Class Mapping: {data_config.get('class_names', ['Unknown', 'Unknown'])}")

# ==========================================
# 3. 预测函数
# ==========================================
inference_transform = transforms.Compose([
    transforms.Resize(data_config.get('img_size', (224, 224))),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=data_config.get('mean', [0.485, 0.456, 0.406]),
        std=data_config.get('std', [0.229, 0.224, 0.225])
    )
])


def predict_image(image_path):
    try:
        # 打开图片
        img = Image.open(image_path).convert("RGB")

        # 预处理
        img_tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

        # 推理
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(prob, dim=1)

        # 获取结果
        class_names = data_config.get('class_names', ['Real', 'AI'])
        pred_label = class_names[pred_idx.item()]
        conf_score = confidence.item()

        print("-" * 30)
        print(f"Image: {image_path}")
        print(f"Prediction: {pred_label}")
        print(f"Confidence: {conf_score:.2%}")
        print("-" * 30)

        return pred_label, conf_score

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

