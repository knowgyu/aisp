# Vision 코드 실습 정답·해설지

이 문서는 생성된 시험 대비 실습 노트북의 `## 정답 입력` 셀에 대응한다.
정답을 바로 복사하기보다 먼저 입력·실행하고, 실패 원인을 기록한 뒤 비교한다.

## 출제 포인트 기준

- 주요 layer 구성과 `[B, C, H, W]` tensor 흐름
- 모델 출력과 정답을 loss/metric으로 연결하는 과정
- 이미지 transform과 학습·평가 데이터 구성

## Vision Practice 01. ResNet18 CIFAR-10 코드 학습

원본: `vision/01_ResNet18_CIFAR10.ipynb`
실습본: `practice_notebooks/vision/01-resnet18-cifar10.ipynb`

### Drill 1 — 3) 학습 유틸리티

원본 Cell `010`. 이 셀은 **3) 학습 유틸리티** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
from dataclasses import dataclass  # 설정을 깔끔하게 묶기 위한 dataclass

@dataclass
class TrainConfig:
    base_epochs_if_new: int = 20  # 처음 실험 시 epoch 수
    extra_epochs_if_resume: int = 10  # 체크포인트 존재 시 추가 실험 epoch 수
    lr: float = 0.1  # learning rate
    weight_decay: float = 5e-4  # L2 정규화
    momentum: float = 0.9  # SGD momentum
    label_smoothing: float = 0.0  # 필요 시 label smoothing, 예) [0, 1, 0] ->  [0.05, 0.9, 0.05]

cfg = TrainConfig()  # config 인스턴스 생성

def accuracy_top1(logits, targets):
    preds = logits.argmax(dim=1)  # 가장 큰 logit의 클래스 선택, logits = [B, C], preds = [B]
    return (preds == targets).float().mean().item()  # top-1 accuracy, scalar

scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())  # AMP용 scaler, 학습 속도를 높이고 메모리 사용량을 줄임
print(scaler)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()  # train mode(dropout/bn 동작 변경)
    total_loss, total_acc = 0.0, 0.0  # 누적 loss/acc
    n = 0  # 샘플 수 누적
    t0 = time.time()  # 시간 측정 시작
    for images, labels in loader:
        images = images.to(device, non_blocking=True)  # 입력을 GPU로 이동 (Asynchronous transfer)
        labels = labels.to(device, non_blocking=True)  # 라벨을 GPU로 이동 (Asynchronous transfer)

        optimizer.zero_grad(set_to_none=True)  # gradient 초기화
        with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):  # AMP autocast
            logits = model(images)  # forward
            loss = criterion(logits, labels)  # loss 계산
        scaler.scale(loss).backward()  # scaled backward
        scaler.step(optimizer)  # optimizer step
        scaler.update()  # scaler 업데이트

        bs = images.size(0)  # batch size
        total_loss += loss.item() * bs  # batch loss 누적
        total_acc  += accuracy_top1(logits.detach(), labels) * bs  # logits값만 복사, batch acc 누적
        n += bs  # 샘플 수 누적

    dt = time.time() - t0  # epoch 수행 시간
    return total_loss / n, total_acc / n, dt  # 평균 loss/acc/시간

@torch.no_grad()   # Autograd off, 함수 내부에서 기울기 계산 멈춤
def evaluate(model, loader, criterion):
    model.eval()  # eval mode, i.e., Dropout 비활성화, BN Running Stats 고정
    total_loss, total_acc = 0.0, 0.0  # 누적
    n = 0  # 샘플 수
    for images, labels in loader:
        images = images.to(device, non_blocking=True)  # GPU 이동
        labels = labels.to(device, non_blocking=True)  # GPU 이동
        logits = model(images)  # forward
        loss = criterion(logits, labels)  # loss
        bs = images.size(0)  # batch size
        total_loss += loss.item() * bs  # 누적
        total_acc  += accuracy_top1(logits, labels) * bs  # 누적
        n += bs  # 누적
    return total_loss / n, total_acc / n  # 평균 loss/acc

def plot_history(hist):
    epochs = np.arange(1, len(hist['train_loss'])+1)  # epoch index
    plt.figure(figsize=(12,4))
    plt.plot(epochs, hist['train_loss'], label='train_loss')  # train loss
    plt.plot(epochs, hist['val_loss'], label='val_loss')  # val loss
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(epochs, hist['train_acc'], label='train_acc')  # train acc
    plt.plot(epochs, hist['val_acc'], label='val_acc')  # val acc
    plt.xlabel('epoch'); plt.ylabel('acc'); plt.legend(); plt.grid(True)
    plt.show()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 7) Activation Map 시각화 (채널별)

원본 Cell `018`. 이 셀은 **7) Activation Map 시각화 (채널별)** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    cur = model  # 시작 모듈
    for part in name.split('.'):  # 점(.) 기준으로 순회
        if part.isdigit():
            cur = cur[int(part)]  # Sequential 인덱스 접근
        else:
            cur = getattr(cur, part)  # attribute 접근
    return cur

def capture_activation(model, layer_name, x_single):
    layer = get_module_by_name(model, layer_name)  # target layer
    activ = {}  # 저장용 dict
    def hook(m, i, o):  # callback function
        activ['feat'] = o.detach().cpu()  # (B,C,H,W) 저장

    h = layer.register_forward_hook(hook)  # target layer처리시 forward hook 등록 (hook 함수 실행)
    model.eval()  # eval 모드
    with torch.no_grad():
        _ = model(x_single)  # forward 수행
    h.remove()  # hook 제거
    assert 'feat' in activ, f"Hook failed for layer {layer_name}"  # 방어 코드
    return activ['feat'][0]  # (C,H,W) 반환

def show_topk_activation_grid(feat_chw, title, topk=16):
    ch_scores = feat_chw.abs().mean(dim=(1,2))  # 채널별 평균 크기 점수
    idx = torch.topk(ch_scores, k=min(topk, feat_chw.shape[0])).indices  # Top-K 채널 인덱스
    maps = []  # (K,1,H,W)로 만들 리스트
    for ci in idx:
        m = feat_chw[int(ci)]  # (H,W)
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)  # 0~1 정규화
        maps.append(m[None, None, ...])  # (1,1,H,W)
    maps = torch.cat(maps, dim=0)  # (K,1,H,W)
    grid = torchvision.utils.make_grid(maps, nrow=8, padding=2)  # grid 생성(종종 3채널로 변환되기도 함)
    plt.figure(figsize=(12,5))
    if grid.ndim == 3 and grid.shape[0] in (3, 4):  # (3,H,W)면 HWC로 변환
        plt.imshow(grid.permute(1,2,0))
    else:  # (1,H,W)면 2D로 표시
        plt.imshow(grid.squeeze(0), cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()


# 1. 시각화할 레이어 이름 정의 (ResNet 구조 기준)
target_layers = {
    "Layer 1": "layer1.1.conv2", # 얕은 레이어
    "Layer 2": "layer2.1.conv2",
    "Layer 3": "layer3.1.conv2",
    "Layer 4": "layer4.1.conv2"  # 깊은 레이어
}

# 2. 입력 이미지 선택(원하면 idx 변경)
idx = 10  # 샘플 인덱스
x, y = test_set[idx]   # 샘플 로드
x_in = x.unsqueeze(0).to(device)

img0 = denorm(x.unsqueeze(0)).squeeze(0).clamp(0,1)
plt.figure(figsize=(3,3))
plt.imshow(img0.permute(1,2,0))
plt.title(f'Input (GT={class_names[y]})')
plt.axis('off')
plt.show()

# 3. 반복문을 통해 각 레이어의 Activation 추출 및 시각화
print(f"{'Layer Name':<20} | {'Activation Shape'}")
print("-" * 45)

for title, layer_name in target_layers.items():
    # Activation 캡처
    feat = capture_activation(model, layer_name, x_in)

    # 정보 출력
    print(f"{layer_name:<20} | {tuple(feat.shape)}") # (C,H,W)

    # 시각화 (Top-16 채널)
    show_topk_activation_grid(feat, f'Activation maps (Top-16) @ {title} ({layer_name})', topk=16)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 9) Grad-CAM 시각화 — Top-K 정분류/오분류 (원본 + Overlay)

원본 Cell `022`. 이 셀은 **9) Grad-CAM 시각화 — Top-K 정분류/오분류 (원본 + Overlay)** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model  # 대상 모델
        self.target_layer = target_layer  # CAM을 뽑을 레이어
        self.activ = None  # forward activation 저장
        self.grad = None  # backward gradient 저장
        self.h1 = target_layer.register_forward_hook(self._forward_hook)  # forward hook
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)  # backward hook

    def _forward_hook(self, module, inp, out):
        self.activ = out  # (B,C,H,W) activation

    def _backward_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0]  # (B,C,H,W) gradient

    def remove(self):
        self.h1.remove()  # hook 제거
        self.h2.remove()  # hook 제거

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        self.model.zero_grad(set_to_none=True)  # grad 초기화
        logits = self.model(x)  # forward
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())  # 예측 클래스 사용
        score = logits[:, class_idx].sum()  # 해당 클래스 점수
        score.backward(retain_graph=False)  # backward로 grad 얻기

        w = self.grad.mean(dim=(2,3), keepdim=True)  # 채널별 중요도(gradient 평균)
        cam = (w * self.activ).sum(dim=1, keepdim=True)  # 중요도 가중합 [B,H, W]
        cam = F.relu(cam)  # ReLU로 양의 기여만
        cam = cam - cam.min()  # min shift
        cam = cam / (cam.max() + 1e-6)  # 0~1 정규화
        return cam.detach(), logits.detach(), class_idx  # cam/logits/class 반환 [B, 1, H, W] / [B, K] / 1

def overlay_heatmap_on_image(img_chw: torch.Tensor, heat_hw: torch.Tensor, alpha=0.45):
    img = img_chw.permute(1,2,0).cpu().numpy()  # CHW -> HWC
    heat = heat_hw.cpu().numpy()  # heatmap to numpy
    cmap = plt.get_cmap('jet')  # 컬러맵
    heat_rgb = cmap(heat)[...,:3]  # heatmap을 RGB로
    out = (1-alpha)*img + alpha*heat_rgb  # overlay
    return np.clip(out, 0, 1)  # [0,1]로 clip

@torch.no_grad()
def predict_all_with_conf(model, loader):
    model.eval()  # eval 모드
    preds, confs, labels = [], [], []  # 누적 리스트
    for x, y in loader:
        x = x.to(device)  # GPU 이동
        y = y.to(device)  # GPU 이동
        p, c, _ = predict_batch(model, x)  # 예측
        preds.append(p.cpu())  # CPU 누적
        confs.append(c.cpu())  # CPU 누적
        labels.append(y.cpu())  # CPU 누적
    return torch.cat(preds), torch.cat(confs), torch.cat(labels)  # concat

# Top-K 선택(정분류/오분류 각각)
K = 8  # 묶음 크기(원하면 변경)
preds_all2, confs_all2, labels_all2 = predict_all_with_conf(model, test_loader)  # test 전체 예측
correct_mask = (preds_all2 == labels_all2)  # 정분류 마스크
wrong_mask = ~correct_mask  # 오분류 마스크

corr_idx = correct_mask.nonzero(as_tuple=False).squeeze(1)  # 정분류 인덱스
corr_sorted = corr_idx[torch.argsort(confs_all2[corr_idx], descending=True)]  # conf 내림차순 정렬
corr_top = corr_sorted[:K]  # Top-K 정분류

wrong_idx = wrong_mask.nonzero(as_tuple=False).squeeze(1)  # 오분류 인덱스
wrong_sorted = wrong_idx[torch.argsort(confs_all2[wrong_idx], descending=True)]  # conf 내림차순 정렬
wrong_top = wrong_sorted[:K]  # Top-K 오분류

print('Top-K correct indices:', corr_top.tolist())  # 출력
print('Top-K wrong indices:', wrong_top.tolist())  # 출력

# Grad-CAM 레이어(일반적으로 마지막 conv 추천)
cam_layer_name = 'layer4.1.conv2'  # CAM 대상 레이어
cam_layer = get_module_by_name(model, cam_layer_name)  # 레이어 객체 얻기
gcam = GradCAM(model, cam_layer)  # GradCAM 인스턴스

def gradcam_for_indices(indices, title):
    overlays = []  # overlay 이미지 리스트(CHW tensor)
    originals = []  # 원본 이미지 리스트(CHW tensor)
    captions = []  # 텍스트 정보 리스트

    for idx in indices.tolist():
        x, y = test_set[idx]  # 원본 샘플(단일)
        x_in = x.unsqueeze(0).to(device)  # 배치화 + GPU 이동

        cam, logits, used_class = gcam(x_in, class_idx=None)  # CAM 계산(예측 클래스 기준)
        prob = F.softmax(logits, dim=1)[0, used_class].item()  # 예측 확률
        pred_name = class_names[used_class]  # 예측 클래스명
        gt_name = class_names[y]  # GT 클래스명

        img0 = denorm(x.unsqueeze(0)).squeeze(0).clamp(0,1)  # 원본(역정규화)
        originals.append(img0.cpu())  # 원본 저장

        cam_resized = F.interpolate(cam, size=(32,32), mode='bilinear', align_corners=False)[0,0].cpu()  # 입력 크기로 resize
        overlay = overlay_heatmap_on_image(img0, cam_resized, alpha=0.45)  # overlay 생성
        overlays.append(torch.from_numpy(overlay).permute(2,0,1))  # HWC -> CHW 저장

        captions.append(f"GT={gt_name} | P={pred_name}({prob:.2f}) | idx={idx}")  # 캡션 저장

    # (1) 원본 그리드
    orig_grid = torchvision.utils.make_grid(torch.stack(originals, dim=0), nrow=len(indices), padding=2)  # original grid
    plt.figure(figsize=(2.2*len(indices), 3))
    plt.imshow(orig_grid.permute(1,2,0))  # 표시
    plt.axis('off')
    plt.title(title + " | Original")
    plt.show()

    # (2) Grad-CAM Overlay 그리드
    over_grid = torchvision.utils.make_grid(torch.stack(overlays, dim=0), nrow=len(indices), padding=2)  # overlay grid
    plt.figure(figsize=(2.2*len(indices), 3))
    plt.imshow(over_grid.permute(1,2,0))  # 표시
    plt.axis('off')
    plt.title(title + f" | Grad-CAM Overlay @ {cam_layer_name}")
    plt.show()

    for c in captions:
        print(c)  # 텍스트로도 출력

gradcam_for_indices(corr_top, f"Top-{K} Correct (High confidence)")  # 정분류 묶음
gradcam_for_indices(wrong_top, f"Top-{K} Wrong (High confidence)")  # 오분류 묶음
gcam.remove()  # hook 해제
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 이 코드의 근사 계산 규칙

원본 Cell `024`. 이 셀은 **이 코드의 근사 계산 규칙** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def rf_update(rf, jump, k, s):
    rf_new = rf + (k - 1) * jump  # receptive field 누적
    jump_new = jump * s  # stride 누적
    return rf_new, jump_new

def estimate_resnet18_cifar_rf():
    rf, jump = 1, 1  # 초기 rf=1, jump=1
    rf, jump = rf_update(rf, jump, k=3, s=1)  # conv1(3x3, stride1)
    for _ in range(2*2):  # layer1: 2 blocks * 2 conv
        rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=2)  # layer2 첫 conv(stride2)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=2)  # layer3 첫 conv(stride2)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=2)  # layer4 첫 conv(stride2)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    rf, jump = rf_update(rf, jump, k=3, s=1)
    return rf, jump

rf, jump = estimate_resnet18_cifar_rf()  # rf/jump 계산
print('Estimated receptive field at last conv stage:', rf)  # rf 출력
print('Effective stride (jump):', jump)  # jump 출력
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## Vision Practice 02. ViT CIFAR-10 코드 학습

원본: `vision/02_ViT_CIFAR10.ipynb`
실습본: `practice_notebooks/vision/02-vit-cifar10.ipynb`

### Drill 1 — 1) 데이터 로딩 & 전처리

원본 Cell `006`. 이 셀은 **1) 데이터 로딩 & 전처리** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import matplotlib.pyplot as plt
import torch

def denormalize_cifar10(img_chw: torch.Tensor):
    """정규화된 CIFAR-10 텐서(C,H,W)를 시각화용(0~1)으로 되돌립니다."""
    mean = torch.tensor(CIFAR10_MEAN, device=img_chw.device)[:, None, None]
    std  = torch.tensor(CIFAR10_STD,  device=img_chw.device)[:, None, None]
    x = img_chw * std + mean
    return x.clamp(0, 1)

def show_batch(data_loader, max_images: int = 16, nrow: int = 4, figsize=(10, 10)):
    """배치에서 일부 샘플을 보기 좋게 시각화합니다.

    - 기본값은 16장(4×4)을 비교적 크게 보여주는 설정입니다.
    - max_images/nrow/figsize를 바꾸면 '더 적게/더 많이', '더 크게/더 작게'를 조절할 수 있습니다.

    - CIFAR-10은 (C,H,W)=(3,32,32) RGB이므로, 시각화할 때는 (H,W,C)로 바꿔야 합니다.
    - make_grid 결과가 figure 안에서 '밀려 보이는' 경우가 있어, subplot 여백을 0으로 맞춰 꽉 차게 표시합니다.

    Args:
        data_loader: DataLoader
        max_images: 한 번에 보여줄 이미지 개수(기본 16장)
        nrow: grid 한 줄에 배치할 이미지 개수(기본 4개)
        figsize: figure 크기
    """
    batch = next(iter(data_loader))
    images, labels = batch  # images: (B,C,H,W)
    images = images[:max_images]

    # make_grid는 (B,C,H,W) → grid(C,H,W)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)

    # 시각화는 정규화 해제 후 HWC로 변환
    grid = denormalize_cifar10(grid).detach().cpu().numpy().transpose((1, 2, 0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, interpolation="nearest")
    ax.set_title(f"Batch Samples (CIFAR-10) | shown={len(images)}")
    ax.axis("off")

    # 여백 제거: grid가 4×4에 꽉 차도록
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()

# 미리보기: 크게(figure) + 적게(16장) 보기
show_batch(train_loader, max_images=16, nrow=4, figsize=(10, 10))
show_batch(train_loader, max_images=16, nrow=4, figsize=(10, 10))
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 2) ViT 모델 구성 개요

원본 Cell `008`. 이 셀은 **2) ViT 모델 구성 개요** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def pair(t):  # image_size/patch_size를 (H,W) 형태로 통일하는 유틸
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):  # Transformer의 Pre-LN 구조: LayerNorm 후 블록 실행
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 토큰 임베딩 차원(dim) 기준 LayerNorm -> 패치별 정규화
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 정규화된 토큰을 Attention/FFN에 전달

class FeedForward(nn.Module):  # Transformer의 FFN(MLP) 블록 -> 패치 자체의 정보를 더 깊게 분석
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(  # FFN: Linear→GELU→Dropout→Linear→Dropout 구성
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):  # 멀티헤드 Self-Attention(MHSA) 구현
    def __init__(self, dim, heads = 4, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 전체 헤드 차원 = head 수 × head 차원
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # 멀티헤드 개수 저장
        self.scale = dim_head ** -0.5  # Scaled dot-product를 위한 스케일(1/√d)

        self.attend = nn.Softmax(dim = -1)  # attention score를 확률로 변환(softmax), 마지막 차원 : sequence length N
        self.last_attn = None  # (학습/추론 시) 마지막 forward에서의 attention map 저장용(시각화/분석)  # attention score를 확률로 변환(softmax), 마지막 차원 : sequence length N
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 입력 토큰→Q,K,V를 한 번에 선형 변환

        self.to_out = nn.Sequential(  # 헤드들을 합친 뒤 출력 투영 + 드롭아웃
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads   # x = [batch Size, tokens, embedding dimension]
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 선형변환 결과를 Q,K,V로 분할 -> (Q, K ,V)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # multi-head 연산을 위해 각 head 기준 n x d 로 분리

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # 각 토큰 간 유사도(Q·Kᵀ) 계산 후 스케일 적용

        attn = self.attend(dots)  # 토큰 간 가중치(attention map) 생성
        self.last_attn = attn.detach()  # 그래프에서 분리(detach)해서 저장 (시각화 목적)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # attention 가중합으로 새로운 토큰 표현(out) 계산
        out = rearrange(out, 'b h n d -> b n (h d)')  # 헤드 차원을 다시 합쳐 (batch, tokens, dim)로 복원
        return self.to_out(out)  # 최종 투영을 거쳐 Attention 블록 출력 반환

class Transformer(nn.Module):  # Encoder block을 여러 층(depth) 쌓는 Transformer
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 각 층(Attention+FFN)을 담을 컨테이너
        for _ in range(depth):  # depth 만큼 Encoder block 반복 생성
            self.layers.append(nn.ModuleList([  # 한 층 = (PreNorm+Attention) + (PreNorm+FFN)
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # Residual 연결: Attention 결과를 입력에 더해 정보 보존
            x = ff(x) + x  # Residual 연결: FFN 결과를 입력에 더해 정보 보존
        return x
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 2-1) ViTConfig + ViT(모델 본체) 구현

원본 Cell `010`. 이 셀은 **2-1) ViTConfig + ViT(모델 본체) 구현** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
from dataclasses import dataclass  # 설정값을 묶어 init 인자를 단순화하기 위한 도구
from typing import Tuple, Union  # 파이썬 버전 호환을 위한 타입 힌트

@dataclass
class ViTConfig:
    # 입력/토큰화 관련
    image_size: 'Union[int, Tuple[int, int]]'          # 입력 이미지 크기 (H, W) 또는 정수(정수면 정사각형)
    patch_size: 'Union[int, Tuple[int, int]]'          # 패치 크기 (Ph, Pw) 또는 정수
    channels: int = 3                          # 입력 채널 수 (CIFAR-10=3, MNIST=1)

    # 모델 본체(Encoder) 관련
    dim: int = 64                              # 토큰 임베딩 차원(Transformer의 hidden size)
    depth: int = 6                             # Encoder 블록 개수(= Transformer layer 수)
    heads: int = 4                             # Multi-Head Self-Attention의 head 개수
    dim_head: int = 64                         # 각 head의 Q/K/V 차원
    mlp_dim: int = 128                         # FFN(MLP) 중간 차원

    # 분류/출력 관련
    num_classes: int = 10                      # 분류 클래스 수
    pool: str = "cls"                          # "cls": CLS 토큰 사용, "mean": 토큰 평균 풀링
    dropout: float = 0.0                       # Encoder 내부 dropout(Attention/FFN)
    emb_dropout: float = 0.0                   # 패치+포지션 임베딩 단계 dropout


class ViT(nn.Module):
    """Vision Transformer (ViT)
    - 이미지를 패치로 쪼개 토큰 시퀀스를 만들고
    - CLS 토큰 + 위치 임베딩을 더한 뒤
    - Transformer Encoder로 전역(Self-Attention) 관계를 학습하여
    - CLS(또는 mean pool) 표현으로 분류합니다.
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()

        # 1) 입력 해상도/패치 크기를 (H,W) 튜플로 정규화
        image_height, image_width = pair(cfg.image_size)
        patch_height, patch_width = pair(cfg.patch_size)

        # 2) 패치가 이미지에 딱 나누어 떨어져야 (h, w) 그리드가 정확히 형성됨
        assert image_height % patch_height == 0 and image_width % patch_width == 0, "Image dimensions must be divisible by the patch size."

        # 3) 패치 개수(=토큰 개수)와 한 패치의 펼친 차원 계산
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = cfg.channels * patch_height * patch_width

        # 4) 풀링 방식 검증: CLS 토큰을 쓸지 mean pool을 쓸지 선택
        assert cfg.pool in {"cls", "mean"}, "pool must be 'cls' or 'mean'"

        # 5) 패치 토큰화: (B,C,H,W) -> (B, N, patch_dim) -> (B, N, dim)
        self.to_patch_embedding = nn.Sequential(
            # 패치 그리드로 자른 뒤, 각 패치를 1D 벡터로 펼쳐 토큰 시퀀스를 만듦
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            # 펼친 패치 벡터를 Transformer hidden size(dim)로 선형 투영
            nn.Linear(patch_dim, cfg.dim),
        )

        # 6) CLS 토큰 + 위치 임베딩(학습 파라미터)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.dim))
        self.dropout = nn.Dropout(cfg.emb_dropout)

        # 7) Transformer Encoder 스택(Attention + FFN + Residual/PreNorm)
        self.transformer = Transformer(
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
        )

        self.pool = cfg.pool

        # 8) 분류 헤드: (CLS/mean) 표현 -> LayerNorm -> Linear(logits)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.num_classes),
        )

    def forward(self, img):
        # A) 패치 임베딩으로 토큰 시퀀스 생성: (B,C,H,W) -> (B,N,dim)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # B) CLS 토큰을 배치만큼 복제해 시퀀스 맨 앞에 붙임: (B,1,dim)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, dim)

        # C) 위치 임베딩을 더해 토큰 순서(공간 위치) 정보를 주입
        x = x + self.pos_embedding[:, : (n + 1)]   # (B, N+1, dim) + (1, N+1, dim) -> Broadcasting
        x = self.dropout(x)

        # D) Encoder를 통과하며 전역 의존성(Self-Attention) 학습
        x = self.transformer(x)  # (B, N+1, dim)

        # E) 이미지 표현 벡터 선택: CLS 토큰(0번) 또는 mean pooling
        x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)   # (B, dim)

        # F) 최종 logits 출력 (softmax는 CrossEntropyLoss 내부에서 처리)
        return self.mlp_head(x)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 6) Testing (Attention)

원본 Cell `024`. 이 셀은 **6) Testing (Attention)** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# (실습) Testing (Attention): ViT가 '어디를 보고' 예측했는지(=CLS→Patch Attention) 시각화
# - 마지막 logits는 CLS 토큰에서 나오므로, "CLS 토큰이 어떤 패치 토큰들을 참고했는지"를 보면 모델의 단서를 얻을 수 있습니다.
# - 실무/연구 관점에서도 초기 레이어(layer_idx=0)는 저수준/분산된 패턴이 많아 해석이 애매한 경우가 많아서,
#   기본값은 "맨 마지막 Transformer 블록"의 attention을 보는 쪽이 직관적입니다.
# - 또한 'head 평균'뿐 아니라 '각 head'가 서로 다른 패턴을 볼 수 있으므로, 여러 head를 한 번에 오버레이해서 비교합니다.

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import random


def denormalize_cifar10_chw(img_chw: torch.Tensor):
    """정규화된 CIFAR-10 이미지(C,H,W)를 시각화용(0~1)으로 복원합니다."""
    mean = torch.tensor(CIFAR10_MEAN, device=img_chw.device)[:, None, None]
    std  = torch.tensor(CIFAR10_STD,  device=img_chw.device)[:, None, None]
    x = img_chw * std + mean
    return x.clamp(0, 1)


def chw_to_hwc_np(img_chw: torch.Tensor) -> np.ndarray:
    """(C,H,W) 텐서를 matplotlib imshow용 (H,W,C) numpy로 변환합니다.
    - CIFAR-10은 정규화가 적용되어 있으므로 denormalize 후 표시합니다.
    """
    x = denormalize_cifar10_chw(img_chw)  # (C,H,W), 0~1
    x = x.detach().cpu()
    # (C,H,W) -> (H,W,C)
    x_hwc = x.permute(1, 2, 0).numpy()
    return x_hwc

def imshow_chw(ax, img_chw: torch.Tensor, title: str = None):
    """CHW 텐서를 안전하게 imshow합니다."""
    x_hwc = chw_to_hwc_np(img_chw)
    ax.imshow(x_hwc)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)


def _extract_last_attn(model, layer_idx: int):
    """지정한 encoder 블록(layer)의 Attention 모듈이 저장해둔 last_attn을 꺼냅니다.
    - Attention.forward에서 self.last_attn = attn.detach()로 저장하도록 구현되어 있어야 합니다.
    - 반환 shape: [B, H, N, N] (B=batch, H=head, N=tokens=1+num_patches)
    """
    # transformer.layers[layer_idx] = (PreNorm(Attention), PreNorm(FFN)) 형태를 가정
    attn_module = model.transformer.layers[layer_idx][0].fn  # [0]:Attention, [1]: FF, PreNorm(fn=Attention)
    attn = attn_module.last_attn
    if attn is None:
        raise RuntimeError("Attention.last_attn이 비어 있습니다. 먼저 model(x)를 한 번 forward 해야 합니다.")
    return attn  # [B,H,N,N]

def _cls_to_patch_map(attn_bhNN, head_idx=None):
    """[B,H,N,N] attention에서 CLS→Patch attention만 뽑아 1D로 반환합니다.
    - CLS 토큰 index = 0
    - Patch 토큰 index = 1..N
    - head_idx=None이면 head 평균을 사용합니다.
    반환: cls_attn_1d shape = [num_patches]
    """
    # CLS(0) 가 각 토큰(j)에 주는 attention: attn[..., 0, j]
    if head_idx is None:
        # head 평균: [B,N] (여기서는 B=1 가정)
        cls_attn = attn_bhNN.mean(dim=1)[0, 0, 1:]  # 평균 관심지도 [1, H, N+1, N+1] → [1, N+1, N+1] → [N]
    else:
        cls_attn = attn_bhNN[0, head_idx, 0, 1:]    # 특정헤드 관심지도 [N]
    return cls_attn

def _to_2d_grid(cls_attn_1d, grid_h: int, grid_w: int):
    """[num_patches] 1D attention을 (grid_h, grid_w)로 reshape합니다."""
    attn_2d = cls_attn_1d.reshape(grid_h, grid_w)
    # 0~1 범위로 정규화(시각화 안정성)
    attn_2d = attn_2d - attn_2d.min()
    attn_2d = attn_2d / (attn_2d.max() + 1e-8)
    return attn_2d

def _upsample_to_image(attn_2d, out_h: int, out_w: int):
    """(grid_h,grid_w) attention을 (out_h,out_w)로 bilinear upsample합니다."""
        # attn_2d가 이미 Tensor일 수도 있으므로, 불필요한 tensor(...) 재구성을 피합니다.
    if isinstance(attn_2d, torch.Tensor):
        attn_t = attn_2d.detach().float().unsqueeze(0).unsqueeze(0)  # [1,1,gh,gw] -> gray image
    else:
        attn_t = torch.as_tensor(attn_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,gh,gw] -> gray image
    attn_up = F.interpolate(attn_t, size=(out_h, out_w), mode="bilinear", align_corners=False)[0,0] # [32, 32]
    attn_up = attn_up - attn_up.min()
    attn_up = attn_up / (attn_up.max() + 1e-8)
    return attn_up.cpu().numpy()  # (out_h,out_w)

def _get_attention_maps_for_image(model, x1, vit_cfg, layer_idx=None, head_idx=None):
    """단일 이미지(x1)에 대해 attention map(overlay용)을 준비합니다.
    - layer_idx=None 또는 layer_idx==0 이면 '맨 마지막 레이어'로 자동 설정합니다(해석 용이).
    - head_idx=None 이면 'head 평균'을 반환.
    - head_idx가 정수면 해당 head만 반환.
    반환: (maps, used_layer_idx)
      - maps: list of (title, attn_up_HxW)
    """
    model.eval()
    with torch.no_grad():
        _ = model(x1)

    # (중요) 기본값: 마지막 레이어를 사용
    depth = int(vit_cfg.depth)
    used_layer_idx = (depth - 1) if (layer_idx is None or layer_idx == 0) else int(layer_idx)

    attn_bhNN = _extract_last_attn(model, used_layer_idx)  # [1,H,N,N]

    # patch grid 크기 = (H/patch, W/patch)
    ps = int(vit_cfg.patch_size) if not isinstance(vit_cfg.patch_size, tuple) else int(vit_cfg.patch_size[0])
    # 입력 이미지 크기 (CIFAR-10: 32x32, MNIST: HxW 등)에서 patch grid 크기를 계산합니다.
    if isinstance(vit_cfg.image_size, tuple):
        img_h, img_w = int(vit_cfg.image_size[0]), int(vit_cfg.image_size[1])
    else:
        img_h, img_w = int(vit_cfg.image_size), int(vit_cfg.image_size)

    grid_h, grid_w = img_h // ps, img_w // ps

    maps = []
    if head_idx is None:
        # head 평균 1장
        cls_1d = _cls_to_patch_map(attn_bhNN, head_idx=None)
        attn_2d = _to_2d_grid(cls_1d, grid_h, grid_w)
        maps.append(("MEAN", _upsample_to_image(attn_2d, out_h=img_h, out_w=img_w)))
    else:
        # 특정 head 1장
        cls_1d = _cls_to_patch_map(attn_bhNN, head_idx=int(head_idx))
        attn_2d = _to_2d_grid(cls_1d, grid_h, grid_w)
        maps.append((f"H{int(head_idx)}", _upsample_to_image(attn_2d, out_h=img_h, out_w=img_w)))

    return maps, used_layer_idx

def show_attention_overlays(
    model,
    test_set,
    vit_cfg,
    device,
    n_samples: int = 6,
    layer_idx=None,
    head_idx=None,
    show_all_heads: bool = True,
    sample_mode: str = "correct",  # "correct" | "incorrect" | "all"
    seed: int = 0,
):
    """여러 이미지에 대해 ORG(원본) + attention overlay를 나란히 보여줍니다.

    Args:
        model: 학습된 ViT 모델
        test_set: (image, label) 형태의 dataset
        vit_cfg: ViTConfig (depth, heads, patch_size 등)
        device: torch.device
        n_samples: 출력할 샘플 개수
        layer_idx: 보고 싶은 encoder 블록 인덱스
            - None 또는 0이면 마지막 레이어(depth-1)로 자동 설정(권장)
        head_idx: 보고 싶은 head 인덱스
            - None이면 head 평균(MEAN)
        show_all_heads: True면 마지막 레이어의 모든 head(H0..H{heads-1})도 함께 출력
        sample_mode: "correct"/"incorrect"/"all" 로 샘플 필터링
        seed: 랜덤 샘플링 재현성

    출력:
        각 샘플마다 [ORG] + [MEAN] + [H0..] 형태로 한 줄(row)로 표시합니다.
    """
    random.seed(seed)

    # 1) 샘플 후보 인덱스 준비
    indices = list(range(len(test_set)))

    # 2) 필요 시: 샘플 필터링 (correct/incorrect/all)
    # sample_mode에 따라 후보를 거릅니다.
    # - "correct"   : 정답(=맞춘) 샘플만
    # - "incorrect" : 오답(=틀린) 샘플만
    # - "all"       : 전체(필터 없음)
    sample_mode = (sample_mode or "all").lower().strip()
    if sample_mode not in {"correct", "incorrect", "all"}:
        print(f"[warn] sample_mode={sample_mode} 는 지원하지 않습니다. -> 'all'로 처리합니다.")
        sample_mode = "all"

    if sample_mode in {"correct", "incorrect"}:
        filtered = []
        model.eval()
        with torch.no_grad():
            for idx in indices:
                x, y = test_set[idx]
                x1 = x.unsqueeze(0).to(device)  # (1, 3, 32, 32)
                logits = model(x1)
                pred = int(logits.argmax(dim=1).item())
                is_correct = (pred == int(y))
                if (sample_mode == "correct" and is_correct) or (sample_mode == "incorrect" and (not is_correct)):
                    filtered.append(idx)

        indices = filtered
        if len(indices) == 0:
            msg = "정답" if sample_mode == "correct" else "오답"
            print(f"[warn] 조건에 맞는 {msg} 샘플이 없습니다. (모델 학습 상태/seed/n_samples를 확인하세요.)")
            return

    # 3) 샘플링
    chosen = random.sample(indices, k=min(n_samples, len(indices))) if len(indices) > 0 else []

    # 4) 시각화: 샘플마다 ORG + (MEAN + Heads) 를 한 row로 출력
    heads = int(vit_cfg.heads)
    for idx in chosen:
        x, y = test_set[idx]
        label = int(y)

        x1 = x.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(x1)
            pred = int(logits.argmax(dim=1).item())

        # (A) 평균 attention(MEAN)
        mean_maps, used_layer = _get_attention_maps_for_image(
            model=model, x1=x1, vit_cfg=vit_cfg,
            layer_idx=layer_idx, head_idx=None
        )

        # (B) 각 head attention(H0..)
        head_maps = []
        if show_all_heads:
            for h in range(heads):
                hm, _ = _get_attention_maps_for_image(
                    model=model, x1=x1, vit_cfg=vit_cfg,
                    layer_idx=layer_idx, head_idx=h
                )
                head_maps.extend(hm)  # hm은 1개짜리 리스트

        # 표시할 컬럼 구성: ORG + MEAN + heads...
        maps_to_show = mean_maps + head_maps

        ncols = 1 + len(maps_to_show)
        fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2))

        # ORG(원본)
        ax0 = axes[0]
        imshow_chw(ax0, x, title=None)
        ax0.axis("off")
        ax0.set_title(f"ORG | GT={class_names[int(label)]} / Pred={class_names[int(pred)]}\n(layer={used_layer})")

        # Overlays
        for j, (title, attn_up) in enumerate(maps_to_show, start=1):
            ax = axes[j]
            base = chw_to_hwc_np(x)  # (H,W,3), 0~1
            ax.imshow(base)
            ax.imshow(attn_up, alpha=0.45)  # heatmap overlay (컬러맵 기본값 사용)
            ax.axis("off")
            ax.set_title(f"{title}")

        plt.tight_layout()
        plt.show()

# 사용 예시:
# - 기본값(layer_idx=None): 마지막 레이어의 MEAN + 모든 head를 출력
# - sample_mode="incorrect": 오답 샘플만 골라서 보면 해석이 더 쉬운 경우가 많습니다.
show_attention_overlays(
    model=model,
    test_set=test_set,
    vit_cfg=vit_cfg,
    device=device,
    n_samples=6,
    layer_idx=None,           # None 또는 0이면 마지막 레이어를 사용
    head_idx=None,            # None이면 평균(MEAN)
    show_all_heads=True,      # 모든 head도 함께 보기
    sample_mode="correct",
    seed=0,
)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## Vision Practice 03. DETR 코드 학습

원본: `vision/03_DETR.ipynb`
실습본: `practice_notebooks/vision/03-detr.ipynb`

### Drill 1 — 2) 입력 전처리 및 박스 좌표 후처리

원본 Cell `007`. 이 셀은 **2) 입력 전처리 및 박스 좌표 후처리** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# [DETR] 입력 전처리 + 박스 좌표 후처리 함수
# - DETR은 COCO로 학습된 ResNet backbone을 사용하므로, ImageNet mean/std 정규화를 맞춰야 합니다.
# - 모델 출력 박스는 (cx, cy, w, h) 형식이며 0~1로 정규화되어 있으므로, (xmin, ymin, xmax, ymax)로 변환 후 원본 이미지 크기로 rescale 합니다.

# standard PyTorch mean-std input image normalization
transform = T.Compose([  # 전처리 파이프라인 정의(논문/코드: data augmentation과 정규화)
    T.Resize(800),  # 짧은 변을 800에 맞추는 리사이즈(공식 데모 설정)
    T.ToTensor(),  # PIL → torch.Tensor (C,H,W), 0~1 스케일
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean/std로 정규화(backbone 사전학습 분포와 정합)
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):  # (cx,cy,w,h) → (xmin,ymin,xmax,ymax) 변환
    x_c, y_c, w, h = x.unbind(1)  # 각 성분을 분리: x.shape=(N,4) 가정
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),  # 중심좌표에서 좌상/우하 꼭짓점 계산
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)  # (N,4)로 다시 스택

def rescale_bboxes(out_bbox, size):  # 정규화 박스를 픽셀 좌표로 변환
    img_w, img_h = size  # PIL size = (W,H)
    b = box_cxcywh_to_xyxy(out_bbox)  # 우선 xyxy로 변환
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)  # (W,H,W,H) 스케일로 곱해 픽셀 단위로 확장
    return b
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 9) 내부 텐서 추출을 위한 forward hook

원본 Cell `021`. 이 셀은 **9) 내부 텐서 추출을 위한 forward hook** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# [DETR] 내부 텐서(특징맵/어텐션) 추출을 위한 forward hook
# - 논문 분석 포인트:
#   1) backbone CNN 특징맵(conv_features)
#   2) 마지막 encoder layer의 self-attention(enc_attn_weights)
#   3) 마지막 decoder layer의 cross-attention(dec_attn_weights)
# - PyTorch hook을 등록해 forward 중간 결과를 캡처합니다.

# use lists to store the outputs via up-values
conv_features, enc_attn_weights, dec_attn_weights = [], [], []  # hook 결과를 담을 리스트(클로저 변수) 초기화

hooks = [  # 각 모듈에 forward hook 등록
    model.backbone[-2].register_forward_hook(  # CNN backbone의 특정 stage 출력(feature map) 저장
        lambda self, input, output: conv_features.append(output) # lambda 인자들 : 실행문
    ),
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(  # encoder self-attention의 attention weights(output[1]) 저장
        lambda self, input, output: enc_attn_weights.append(output[1])   # self-attention weight -> output[1]
    ),
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(  # decoder cross-attention의 attention weights(output[1]) 저장
        lambda self, input, output: dec_attn_weights.append(output[1])   # cross-attention weight -> output[1]
    ),
]

# propagate through the model
outputs = model(img)  # hook이 등록된 상태로 forward 실행

for hook in hooks:  # 등록했던 hook 해제(메모리/부작용 방지)
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]  # hook 결과를 담을 리스트(클로저 변수) 초기화
print(f"features shape: {conv_features['0'].tensors.shape}")   # [1, channels, H, W]
enc_attn_weights = enc_attn_weights[0]  # encoder attention weights 텐서
print(f"enc_attn_weights shape: {enc_attn_weights.shape}")   # [1, HW, HW] -> 각 feature간 attention
dec_attn_weights = dec_attn_weights[0]  # decoder attention weights 텐서
print(f"dec_attn_weights shape: {dec_attn_weights.shape}")   # [1, N, HW] -> 각 객체와 feature간 attention
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 10) Decoder cross-attention 시각화

원본 Cell `023`. 이 셀은 **10) Decoder cross-attention 시각화** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# [DETR] 특징맵 shape 확인
# - backbone 출력 feature map의 공간 해상도(H,W)는 입력 대비 downsample된 크기입니다.
# - DETR은 이 feature map을 1D sequence로 펼쳐 transformer encoder에 넣습니다.

# get the feature map shape
h, w = conv_features['0'].tensors.shape[-2:]    # NestedTensor 딕셔너리에서 마지막 stage feature 선택('0' 키)

fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
colors = COLORS * 100
for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].view(h, w))  # keep에 해당하는 객체(idx)의 cross attention map
    ax.axis('off')
    ax.set_title(f'query id: {idx.item()}')
    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color='blue', linewidth=3))
    ax.axis('off')
    ax.set_title(CLASSES[probas[idx].argmax()])
fig.tight_layout()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 14) AttentionVisualizer (Encoder Self-attention) 클래스

원본 Cell `031`. 이 셀은 **14) AttentionVisualizer (Encoder Self-attention) 클래스** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# [DETR] AttentionVisualizer 클래스
# - 목적: 이미지 내 특정 지점(Pixel)을 기준으로 Encoder의 'Self-attention'이 어떻게 전역 문맥을 섞는지 직관적으로 이해.
# - 주요 아이디어(논문 구현 관점):
#   1) Backbone -> Transformer Encoder: 이미지를 토큰화하고 각 토큰(픽셀)들 간의 관계를 Self-attention으로 연산.
#   2) 사용자가 슬라이더로 이미지 상의 특정 좌표(Query 지점)를 선택.
#   3) 해당 좌표의 토큰이 이미지 전체(Key 지점들)를 얼마나 참고하는지 마지막 Encoder 레이어에서 추출하여 시각화.
# - 아래 코드는 forward hook으로 Encoder 내부 텐서를 캡처하고, 인터랙티브 위젯을 통해 특정 위치의 어텐션 맵을 그립니다.

class AttentionVisualizer:  # DETR 내부 어텐션을 시각화하는 유틸 클래스 정의
    def __init__(self, model, transform):  # 모델/전처리/위젯 초기화
        self.model = model  # 외부에서 로드한 DETR 모델 보관
        self.transform = transform  # 입력 이미지 전처리 함수 보관

        self.url = ""
        self.cur_url = None
        self.pil_img = None
        self.tensor_img = None

        self.conv_features = None
        self.enc_attn_weights = None
        self.dec_attn_weights = None

        self.setup_widgets()

    def setup_widgets(self):
        self.sliders = [
            widgets.Text(
                value='http://images.cocodataset.org/val2017/000000039769.jpg',
                placeholder='Type something',
                description='URL (ENTER):',
                disabled=False,
                continuous_update=False,
                layout=widgets.Layout(width='100%')
            ),
            widgets.FloatSlider(min=0, max=0.99,
                        step=0.02, description='X coordinate', value=0.72,
                        continuous_update=False,
                        layout=widgets.Layout(width='50%')
                        ),
            widgets.FloatSlider(min=0, max=0.99,
                        step=0.02, description='Y coordinate', value=0.40,
                        continuous_update=False,
                        layout=widgets.Layout(width='50%')),
            widgets.Checkbox(
              value=False,
              description='Direction of self attention',
              disabled=False,
              indent=False,
              layout=widgets.Layout(width='50%'),
          ),
            widgets.Checkbox(
              value=True,
              description='Show red dot in attention',
              disabled=False,
              indent=False,
              layout=widgets.Layout(width='50%'),
          )
        ]
        self.o = widgets.Output()  # matplotlib 출력이 들어갈 영역

    def compute_features(self, img):  # hook을 등록해 feature/attention을 수집한 뒤 forward 수행
        model = self.model  # 외부에서 로드한 DETR 모델 보관
        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(  # backbone feature map 캡처
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(  # encoder 마지막 layer self-attention 캡처
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(  # decoder 마지막 layer cross-attention 캡처
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
        # propagate through the model
        outputs = model(img)  # hook이 걸린 상태로 DETR 추론 실행

        for hook in hooks:  # hook 해제(중복 캡처 방지)
            hook.remove()

        # don't need the list anymore
        self.conv_features = conv_features[0]  # 리스트에서 실제 텐서/NestedTensor 꺼내기
        self.dec_attn_weights = dec_attn_weights[0]
        # get the HxW shape of the feature maps of the CNN
        shape = self.conv_features['0'].tensors.shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        self.enc_attn_weights = enc_attn_weights[0].reshape(shape + shape)

    def compute_on_image(self, url):
        if url != self.url:
            self.url = url
            self.pil_img = Image.open(requests.get(url, stream=True).raw)
            # mean-std normalize the input image (batch-size: 1)
            self.tensor_img = self.transform(self.pil_img).unsqueeze(0)  # 입력 이미지 전처리 함수 보관
            self.compute_features(self.tensor_img)

    def update_chart(self, change):
        with self.o:  # Output 영역에 그림을 그리기 위해 컨텍스트 진입
            clear_output()  # 이전 출력 삭제 후 새 결과 렌더링

            # j and i are the x and y coordinates of where to look at
            # sattn_dir is which direction to consider in the self-attention matrix
            # sattn_dot displays a red dot or not in the self-attention map
            url, j, i, sattn_dir, sattn_dot = [s.value for s in self.sliders]

            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(9, 4))
            self.compute_on_image(url)

            # convert reference point to absolute coordinates
            j = int(j * self.tensor_img.shape[-1])
            i = int(i * self.tensor_img.shape[-2])

            # how much was the original image upsampled before feeding it to the model
            scale = self.pil_img.height / self.tensor_img.shape[-2]

            # compute the downsampling factor for the model
            # it should be 32 for standard DETR and 16 for DC5
            sattn = self.enc_attn_weights
            fact = 2 ** round(math.log2(self.tensor_img.shape[-1] / sattn.shape[-1]))

            # round the position at the downsampling factor
            x = ((j // fact) + 0.5) * fact
            y = ((i // fact) + 0.5) * fact

            axs[0].imshow(self.pil_img)
            axs[0].axis('off')
            axs[0].add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))

            idx = (i // fact, j // fact)

            if sattn_dir:
                sattn_map = sattn[idx[0], idx[1], ...]
            else:
                sattn_map = sattn[..., idx[0], idx[1]]

            axs[1].imshow(sattn_map, cmap='cividis', interpolation='nearest')  # attention heatmap 출력
            if sattn_dot:
                axs[1].add_patch(plt.Circle((idx[1],idx[0]), 1, color='r'))
            axs[1].axis('off')
            axs[1].set_title(f'self-attention{(i, j)}')

            plt.show()

    def run(self):
      for s in self.sliders:
          s.observe(self.update_chart, 'value')
      self.update_chart(None)
      url, x, y, d, sattn_d = self.sliders
      res = widgets.VBox(
      [
          url,
          widgets.HBox([x, y]),
          widgets.HBox([d, sattn_d]),
          self.o
      ])
      return res
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## Vision Practice 04. U-Net Segmentation 코드 학습

원본: `vision/04_Unet.ipynb`
실습본: `practice_notebooks/vision/04-unet.ipynb`

### Drill 1 — 1) 데이터 준비

원본 Cell `004`. 이 셀은 **1) 데이터 준비** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class SyntheticCircleDataset(Dataset):
    """
    매우 단순한 synthetic 데이터셋 예제입니다.

    - 입력: 1채널 이미지 (H x W)
    - 타깃: 원 모양의 binary mask
    """
    def __init__(self, length=200, image_size=128, transform=None):
        self.length = length
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 빈 이미지 생성
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        mask = np.zeros_like(img)

        # 랜덤 원 생성
        center_x = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)    # 1/4 ~ 3/4
        center_y = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
        radius = np.random.randint(self.image_size // 8, self.image_size // 4)

        y, x = np.ogrid[:self.image_size, :self.image_size]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask[dist_from_center <= radius] = 1.0

        # 입력 이미지는 약간의 노이즈 + mask를 이용해 생성
        img = mask + 0.2 * np.random.randn(self.image_size, self.image_size).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

        # (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        if self.transform is not None:
            # 필요 시 torchvision.transforms 등을 추가적으로 적용
            pass

        return img, mask

# 데이터셋/데이터로더 예시
train_dataset = SyntheticCircleDataset(length=400, image_size=128)
val_dataset   = SyntheticCircleDataset(length=100, image_size=128)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 2) U-Net 모델 구현

원본 Cell `006`. 이 셀은 **2) U-Net 모델 구현** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# =========================
# 3) U-Net 모델 정의
# =========================
# 이 섹션의 목표:
# - U-Net의 "인코더(Down) → 보틀넥 → 디코더(Up) + Skip Connection" 구조를
#   PyTorch 코드로 직접 따라가며 이해합니다.
#
# 핵심 아이디어(한 줄):
# - Down에서 공간 해상도(H,W)는 줄이고 채널(C)은 늘리면서 특징을 추출하고,
#   Up에서 해상도를 복원하면서 Down 단계의 특징맵을 Skip으로 concat하여
#   localization(위치 정보)을 되살립니다.

class DoubleConv(nn.Module):
    """U-Net 기본 블록: (Conv → ReLU) × 2

    - 첫 번째 Conv가 채널을 '중간 채널(mid_channels)'로 바꾸고,
      두 번째 Conv가 '출력 채널(out_channels)'로 맞춥니다.
    - 기본값(mid_channels=None)일 때는 mid_channels=out_channels로 두어,
      (C_in → C_out → C_out) 형태가 됩니다.

    입력/출력 텐서 형태:
      - 입력:  (B, C_in, H, W)
      - 출력:  (B, C_out, H, W)  # padding=1 이라 H,W 유지
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()

        # U-Net에서 업샘플 후 concat을 하면 채널이 2배가 되므로,
        # bilinear 업샘플링을 쓸 때는 mid_channels=in_channels//2 처럼
        # '중간 채널'을 줄여주는 방식이 흔히 사용됩니다.
        if mid_channels is None:
            mid_channels = out_channels

        self.net = nn.Sequential(
            # 1) (C_in → C_mid)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),

            # 2) (C_mid → C_out)
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Down(nn.Module):
    """Downsampling 블록: MaxPool(2)로 해상도 1/2 → DoubleConv

    입력/출력 텐서 형태:
      - 입력:  (B, C_in,  H,  W)
      - 출력:  (B, C_out, H/2, W/2)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 해상도 축소
        x = self.pool(x)           # (B, C_in, H/2, W/2)
        # 2) 채널 확장 + 특징 추출
        x = self.conv(x)           # (B, C_out, H/2, W/2)
        return x


class Up(nn.Module):
    """Upsampling 블록: 업샘플링 → (Skip concat) → DoubleConv

    구현 관점(중요):
    - in_channels 는 concat 이후 채널 수를 의미합니다.
      예) x1(디코더) 채널=512, x2(skip) 채널=512 → concat 채널=1024 → in_channels=1024

    - bilinear=True:
        1) 업샘플은 파라미터 없는 bilinear interpolation으로 수행
        2) concat 후 DoubleConv에서 mid_channels를 in_channels//2 로 두어
           채널을 자연스럽게 '절반'으로 줄이는 방식(원 논문/레퍼런스 구현과 동일 계열)

    - bilinear=False:
        ConvTranspose2d로 업샘플 자체를 학습(파라미터 증가)
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()

        if bilinear:
            # (B, C, H/2, W/2) → (B, C, H, W)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

            # bilinear일 때는 업샘플 후 채널 수(C)가 그대로 유지됩니다.
            # concat 결과(in_channels)를 DoubleConv로 처리하되,
            # 첫 conv의 출력(mid_channels)을 in_channels//2로 두어 채널을 줄입니다.
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            # (B, C, H/2, W/2) → (B, C/2, H, W)  (deconv가 채널도 절반으로 줄여줌)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: (B, C_dec, H/2, W/2)  / x2: (B, C_skip, H, W)

        # 1) 업샘플링: 해상도를 skip과 맞추기
        x1 = self.up(x1)  # bilinear: 채널 유지 / deconv: 채널이 절반으로 감소

        # 2) (필요 시) 패딩으로 크기 정렬
        #    - 홀수 크기 입력 등으로 인해 x1과 x2의 H/W가 1~2 픽셀 정도 다를 수 있습니다.
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,  # left, right
                        diff_y // 2, diff_y - diff_y // 2]) # top, bottom

        # 3) 채널 방향 concat (skip 연결)
        #    cat dim=1 은 채널(C) 축
        x = torch.cat([x2, x1], dim=1)  # (B, C_skip + C_up, H, W) == (B, in_channels, H, W)

        # 4) conv 블록으로 특징 정제 + 채널 축소
        return self.conv(x)

class OutConv(nn.Module):
    """마지막 1x1 conv: 채널을 클래스 수로 매핑

    예)
      - binary segmentation: n_classes=1 (logits 1채널)
      - multi-class:         n_classes=K (logits K채널)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 전체 모델

    전형적인 채널 구성 예:
      1 → 64 → 128 → 256 → 512 → 1024 (down)
      1024 → 512 → 256 → 128 → 64 (up)
    """
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super().__init__()

        # -------------------------
        # Encoder (Contracting path)
        # -------------------------
        self.inc = DoubleConv(n_channels, 64)   # (B, n_channels, H, W) → (B, 64, H, W)
        self.down1 = Down(64, 128)              # → (B, 128, H/2, W/2)
        self.down2 = Down(128, 256)             # → (B, 256, H/4, W/4)
        self.down3 = Down(256, 512)             # → (B, 512, H/8, W/8)

        # bilinear 업샘플링이면 파라미터/연산을 줄이기 위해 bottleneck 채널을 1024 대신 512로 줄이는 경우가 많음
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # → (B, 1024/f, H/16, W/16)

        # -------------------------
        # Decoder (Expanding path)
        # -------------------------
        # Up 블록의 in_channels는 "concat 후 채널" 기준으로 설계되어야 합니다.
        self.up1 = Up(1024, 512 // factor, bilinear)  # (skip=512)와 concat을 고려
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # -------------------------
        # Output head
        # -------------------------
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------------------------
        # Encoder: skip을 위해 각 단계 출력 저장
        # -------------------------
        x1 = self.inc(x)       # (B, 64, H, W)
        x2 = self.down1(x1)    # (B, 128, H/2, W/2)
        x3 = self.down2(x2)    # (B, 256, H/4, W/4)
        x4 = self.down3(x3)    # (B, 512, H/8, W/8)
        x5 = self.down4(x4)    # (B, 1024/f, H/16, W/16)

        # -------------------------
        # Decoder: 업샘플 + skip concat
        # -------------------------
        x = self.up1(x5, x4)   # (B, 512/f, H/8, W/8)
        x = self.up2(x, x3)    # (B, 256/f, H/4, W/4)
        x = self.up3(x, x2)    # (B, 128/f, H/2, W/2)
        x = self.up4(x, x1)    # (B, 64, H, W)

        # -------------------------
        # 최종 logits 출력
        # -------------------------
        logits = self.outc(x)  # (B, n_classes, H, W)
        return logits


# =========================
# =========================
# 4) 모델 생성 및 구조 확인
# =========================
model = UNet(n_channels=1, n_classes=1).to(device)

# -------------------------
# 4) 모델 구조 확인: print(model) + torchinfo.summary
# -------------------------
print(model)

# torchinfo.summary는 forward를 한 번 실행해 각 레이어의 출력 shape/파라미터 수를 정리해줍니다.
# - 모델이 학습 모드(train)일 때 BatchNorm/Dropout 등이 있을 수 있으니,
#   구조 확인 목적에서는 eval()로 두는 것이 안전합니다.
model.eval()

try:
    from torchinfo import summary

    # 입력 크기: (B, C, H, W)
    # - 여기서는 (1, 1, 128, 128) 텐서를 기준으로 요약합니다.
    # - 다른 해상도로 실험하려면 H,W만 바꾸면 됩니다.
    _summary = summary(
        model,
        input_size=(1, 1, 128, 128),
        depth=4,
        device=device,   # cpu/cuda 환경에 맞춰 동일 디바이스에서 실행
    )
    print(_summary)
except Exception as e:
    print("[torchinfo.summary] 실행 불가:", repr(e))
    print("대응:")
    print("  1) torchinfo 설치 확인: pip install torchinfo")
    print("  2) 입력 크기/채널 수가 모델 기대와 맞는지 확인")
    print("  3) 앞선 forward sanity check가 통과하는지 확인")

# -------------------------
# 5) Forward sanity check (shape 확인)
# -------------------------
dummy = torch.randn(1, 1, 128, 128, device=device)
with torch.no_grad():
    out = model(dummy)

print("Input shape:", tuple(dummy.shape), "Output shape:", tuple(out.shape))
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 4) 학습 루프

원본 Cell `010`. 이 셀은 **4) 학습 루프** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = criterion(logits, masks)
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch}/{num_epochs}]  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 5) 결과 시각화

원본 Cell `014`. 이 셀은 **5) 결과 시각화** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
@torch.no_grad()
def visualize_predictions(model, dataloader, device, num_samples=3):
    model.eval()
    imgs, masks = next(iter(dataloader))
    imgs = imgs.to(device)
    masks = masks.to(device)

    logits = model(imgs)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    imgs = imgs.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    num_samples = min(num_samples, imgs.shape[0])

    plt.figure(figsize=(9, num_samples * 3))
    for i in range(num_samples):
        # 입력
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(imgs[i, 0], cmap='gray')
        plt.title('Input')
        plt.axis('off')

        # GT mask
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(masks[i, 0], cmap='gray')
        plt.title('GT Mask')
        plt.axis('off')

        # Pred mask
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(preds[i, 0], cmap='gray')
        plt.title('Pred Mask')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


visualize_predictions(model, val_loader, device, num_samples=3)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?
