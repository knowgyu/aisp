# On-Device AI 코드 실습 정답·해설지

이 문서는 생성된 시험 대비 실습 노트북의 `## 정답 입력` 셀에 대응한다.
정답을 바로 복사하기보다 먼저 입력·실행하고, 실패 원인을 기록한 뒤 비교한다.

## 출제 포인트 기준

- Quantization의 scale/zero-point와 양자화·역양자화
- Pruning 중요도·mask와 연산 제거
- Teacher/Student 출력과 정답을 결합한 distillation loss

## Practice 01. Pruning for CNN 코드 학습

원본: `On-Device AI 강의자료/실습/1. Pruning for CNN.ipynb`
실습본: `practice_notebooks/on_device_ai/01-pruning-cnn.ipynb`

### Drill 1 — CIFAR-10 데이터셋 준비

원본 Cell `007`. 이 셀은 **CIFAR-10 데이터셋 준비** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    "test": Compose([
        ToTensor(),
    ]),
}

# torchvision의 공식 다운로드 서버 에러를 피하기 위해
# Hugging Face Hub에서 CIFAR-10을 로드하는 커스텀 Dataset 클래스
class HFCIFAR10(Dataset):
    def __init__(self, split, transform=None):
        self.data = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform
        self.targets = list(self.data["label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Subset에서 numpy.int64가 들어올 수 있으므로 Python int로 변환
        idx = int(idx)
        item = self.data[idx]
        # Hugging Face CIFAR-10 field names: img, label
        image = item["img"].convert("RGB")
        label = item["label"]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

dataset = {split: HFCIFAR10(split=split, transform=transforms[split])
           for split in ["train", "test"]}

dataloader = {
    split: DataLoader(dataset[split], batch_size=512,
                      shuffle=(split=='train'),
                      num_workers=0, pin_memory=True)
    for split in ['train', 'test']
}
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 모델의 정확도 평가

원본 Cell `013`. 이 셀은 **모델의 정확도 평가** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    callbacks=None,
) -> None:
    model.train()
    for inputs, targets in tqdm(dataloader, desc='Train', leave=False):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if callbacks is not None:
            for callback in callbacks:
                callback()

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
) -> float:
    model.eval()
    num_samples, num_correct = 0, 0
    for inputs, targets in tqdm(dataloader, desc="Eval", leave=False, disable=not verbose):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        num_samples += targets.size(0)
        num_correct += (preds == targets).sum()
    return (num_correct / num_samples * 100).item()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — [실습 1] Fine-grained Pruning 구현

원본 Cell `028`. 이 셀은 **[실습 1] Fine-grained Pruning 구현** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def prune_weight_fine_grained(weight: torch.Tensor, sparsity: float) -> None:
    """가중치 텐서에 대해 fine-grained pruning을 수행하는 함수

    Args:
        weight: pruning할 가중치 텐서
        sparsity: pruning할 비율 (0~1 사이 값)

    Returns:
        pruning mask 텐서
    """
    # sparsity 값을 0~1 사이로 제한
    sparsity = min(1.0, max(0.0, sparsity))

    # 특수한 경우 처리
    if sparsity == 1.0:  # 모든 가중치를 제거
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:  # 모든 가중치를 유지
        return torch.ones_like(weight)

    ##################### YOUR CODE STARTS HERE #####################
    # 제거할 원소 개수를 계산하세요.
    # hint: round() 함수를 사용하세요.
    num_pruned_elements =

    # 가중치의 중요도를 절댓값으로 importance 계산
    # hint: torch.abs() 함수를 사용하세요.
    importance =

    # pruning trheshold를 계산하세요.
    # hint: torch.kthvalue() 함수를 사용하세요.
    threshold =

    # threshold보다 큰 값들은 유지(1), 작은 값들은 제거(0)하는 마스크 생성
    # hint: 부등호를 사용하세요.
    mask =
    ##################### YOUR CODE ENDS HERE #######################

    # 마스크를 적용하여 pruning 수행
    weight.mul_(mask)

    return mask

# 마스크 생성 및 시각화
mask_fine_grained = prune_weight_fine_grained(weight.clone(), prune_sparsity)
draw_weight_distribution(mask_fine_grained, title="Fine-grained Pruning Mask")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — [실습 2] Sensitivity Analysis를 통한 Pruning 수행

원본 Cell `046`. 이 셀은 **[실습 2] Sensitivity Analysis를 통한 Pruning 수행** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class FineGrainedPruner:
    """Fine-grained pruning을 수행하는 클래스"""
    def __init__(self, model, sparsity_dict):
        """
        Args:
            model: pruning할 모델
            sparsity_dict: 레이어별 희소성 비율을 담은 딕셔너리
        """
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        """마스크를 모델에 적용"""
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        """
        모델의 각 레이어에 대해 희소성 비율에 따라 가지치기 수행

        Args:
            model: 가지치기할 모델
            sparsity_dict: 레이어별 희소성 비율을 담은 딕셔너리

        Returns:
            masks: 레이어별 마스크를 담은 딕셔너리
        """
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                masks[name] = prune_weight_fine_grained(param, sparsity_dict[name])
        return masks

def get_uniform_sparsity_dict(sparsity):
    """
    모든 레이어에 동일한 희소성 비율을 적용하는 딕셔너리 생성

    Args:
        sparsity: 적용할 희소성 비율

    Returns:
        sparsity_dict: 레이어별 동일한 희소성 비율을 담은 딕셔너리
    """
    sparsity_dict = {name: sparsity for name, param in model.named_parameters() if param.dim() > 1}
    return sparsity_dict
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — [실습 3] Global Magnitude Pruning 구현

원본 Cell `053`. 이 셀은 **[실습 3] Global Magnitude Pruning 구현** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class FineGrainedPrunerV2:
    def __init__(self, model, sparsity, global_prune=False):
        """
        전역 또는 레이어별 프루닝을 위한 프루너 클래스

        Args:
            model: 프루닝할 모델
            sparsity: 프루닝 비율 (0~1)
            global_prune: 전역 프루닝 여부
        """
        self.masks = FineGrainedPrunerV2.prune(model, sparsity, global_prune)

    @torch.no_grad()
    def apply(self, model):
        """프루닝 마스크를 모델에 적용"""
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity, global_prune):
        """
        전역 또는 레이어별 프루닝 수행

        Args:
            model: 프루닝할 모델
            sparsity: 프루닝 비율 (0~1)
            global_prune: 전역 프루닝 여부

        Returns:
            masks: 프루닝 마스크 딕셔너리
        """
        masks = dict()
        if global_prune:
            # 모든 2D 이상의 파라미터를 1차원으로 변환하여 수집
            parameters_to_prune = []
            for name, param in model.named_parameters():
                if param.dim() > 1:  # conv, fc 레이어만 프루닝
                    parameters_to_prune.append(param.view(-1))

            ##################### YOUR CODE STARTS HERE #####################
            # 모든 weight를 하나의 텐서로 결합해주세요..
            # hint: torch.cat()을 사용하세요.
            all_weights =

            # all_weights를 대상으로 global threshold를 구해주세요.
            num_elements =
            num_zeros =
            importance =
            threshold =
            ##################### YOUR CODE ENDS HERE #######################

            # threshold 기반 마스크 생성
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    mask = torch.abs(param.data) > threshold
                    masks[name] = mask
        else:
            # 레이어별 프루닝 수행
            for name, param in model.named_parameters():
                if param.dim() > 1: # we only prune conv and fc weights
                    masks[name] = prune_weight_fine_grained(param, sparsity)
        return masks
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## Practice 02. Quantization for CNN 코드 학습

원본: `On-Device AI 강의자료/실습/2. Quantization for CNN.ipynb`
실습본: `practice_notebooks/on_device_ai/02-quantization-cnn.ipynb`

### Drill 1 — Setup

원본 Cell `007`. 이 셀은 **Setup** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class VGG(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

  def __init__(self) -> None:
    super().__init__()

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    for x in self.ARCH:
      if x != 'M':
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))
    add("avgpool", nn.AvgPool2d(2))
    self.backbone = nn.Sequential(OrderedDict(layers))
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    # x = x.mean([2, 3])
    x = x.view(x.shape[0], -1)

    # classifier: [N, 512] => [N, 10]
    x = self.classifier(x)
    return x
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 계산을 위해 임시로 이 값을 FP32로 변환할 것입니다.

원본 Cell `070`. 이 셀은 **계산을 위해 임시로 이 값을 FP32로 변환할 것입니다.** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class QuantizedConv2d(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 stride, padding, dilation, groups,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # 현재 버전의 PyTorch는 IntTensor를 nn.Parameter로 지원하지 않습니다.
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth


    def forward(self, x):
        return quantized_conv2d(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale,
            self.stride, self.padding, self.dilation, self.groups
            )

class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # 현재 버전의 PyTorch는 IntTensor를 nn.Parameter로 지원하지 않습니다.
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_linear(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale
            )

class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # 현재 버전의 PyTorch는 정수 기반의 MaxPool 연산을 지원하지 않습니다.
        return super().forward(x.float()).to(torch.int8)

class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # 현재 버전의 PyTorch는 정수 기반의 AvgPool 연산을 지원하지 않습니다.
        return super().forward(x.float()).to(torch.int8)

feature_bitwidth = weight_bitwidth = 8
quantized_model = copy.deepcopy(model_fused)
quantized_backbone = []
ptr = 0
while ptr < len(quantized_model.backbone):
    if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and \
        isinstance(quantized_model.backbone[ptr + 1], nn.ReLU):
        conv = quantized_model.backbone[ptr]
        conv_name = f'backbone.{ptr}'
        relu = quantized_model.backbone[ptr + 1]
        relu_name = f'backbone.{ptr + 1}'

        input_scale, input_zero_point = \
            get_quantization_scale_and_zero_point(
                input_activation[conv_name], feature_bitwidth)

        output_scale, output_zero_point = \
            get_quantization_scale_and_zero_point(
                output_activation[relu_name], feature_bitwidth)

        quantized_weight, weight_scale, weight_zero_point = \
            linear_quantize_weight_per_tensor(conv.weight.data, weight_bitwidth)
        quantized_bias, bias_scale, bias_zero_point = \
            linear_quantize_bias_per_output_channel(
                conv.bias.data, weight_scale, input_scale)
        shifted_quantized_bias = \
            shift_quantized_conv2d_bias(quantized_bias, quantized_weight,
                                        input_zero_point)

        quantized_conv = QuantizedConv2d(
            quantized_weight, shifted_quantized_bias,
            input_zero_point, output_zero_point,
            input_scale, weight_scale, output_scale,
            conv.stride, conv.padding, conv.dilation, conv.groups,
            feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
        )

        quantized_backbone.append(quantized_conv)
        ptr += 2
    elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
        quantized_backbone.append(QuantizedMaxPool2d(
            kernel_size=quantized_model.backbone[ptr].kernel_size,
            stride=quantized_model.backbone[ptr].stride
            ))
        ptr += 1
    elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
        quantized_backbone.append(QuantizedAvgPool2d(
            kernel_size=quantized_model.backbone[ptr].kernel_size,
            stride=quantized_model.backbone[ptr].stride
            ))
        ptr += 1
    else:
        raise NotImplementedError(type(quantized_model.backbone[ptr]))  # should not happen
quantized_model.backbone = nn.Sequential(*quantized_backbone)

# finally, quantized the classifier
fc_name = 'classifier'
fc = model.classifier
input_scale, input_zero_point = \
    get_quantization_scale_and_zero_point(
        input_activation[fc_name], feature_bitwidth)

output_scale, output_zero_point = \
    get_quantization_scale_and_zero_point(
        output_activation[fc_name], feature_bitwidth)

quantized_weight, weight_scale, weight_zero_point = \
    linear_quantize_weight_per_tensor(fc.weight.data, weight_bitwidth)
quantized_bias, bias_scale, bias_zero_point = \
    linear_quantize_bias_per_output_channel(
        fc.bias.data, weight_scale, input_scale)
shifted_quantized_bias = \
    shift_quantized_linear_bias(quantized_bias, quantized_weight,
                                input_zero_point)

quantized_model.classifier = QuantizedLinear(
    quantized_weight, shifted_quantized_bias,
    input_zero_point, output_zero_point,
    input_scale, weight_scale, output_scale,
    feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 계산을 위해 임시로 이 값을 FP32로 변환할 것입니다.

원본 Cell `074`. 이 셀은 **계산을 위해 임시로 이 값을 FP32로 변환할 것입니다.** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class QuantizedConv2d(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 stride, padding, dilation, groups,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # 현재 버전의 PyTorch는 IntTensor를 nn.Parameter로 지원하지 않습니다.
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth


    def forward(self, x):
        return quantized_conv2d(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale,
            self.stride, self.padding, self.dilation, self.groups
            )

class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # 현재 버전의 PyTorch는 IntTensor를 nn.Parameter로 지원하지 않습니다.
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_linear(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale
            )

class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # 현재 버전의 PyTorch는 정수 기반의 MaxPool 연산을 지원하지 않습니다.
        return super().forward(x.float()).to(torch.int8)

class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # 현재 버전의 PyTorch는 정수 기반의 AvgPool 연산을 지원하지 않습니다.
        return super().forward(x.float()).to(torch.int8)

feature_bitwidth = weight_bitwidth = 8
quantized_model = copy.deepcopy(model_fused)
quantized_backbone = []
ptr = 0
while ptr < len(quantized_model.backbone):
    if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and \
        isinstance(quantized_model.backbone[ptr + 1], nn.ReLU):
        conv = quantized_model.backbone[ptr]
        conv_name = f'backbone.{ptr}'
        relu = quantized_model.backbone[ptr + 1]
        relu_name = f'backbone.{ptr + 1}'

        input_scale, input_zero_point = \
            get_quantization_scale_and_zero_point(
                input_activation[conv_name], feature_bitwidth)

        output_scale, output_zero_point = \
            get_quantization_scale_and_zero_point(
                output_activation[relu_name], feature_bitwidth)

        quantized_weight, weight_scale, weight_zero_point = \
            linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)
        quantized_bias, bias_scale, bias_zero_point = \
            linear_quantize_bias_per_output_channel(
                conv.bias.data, weight_scale, input_scale)
        shifted_quantized_bias = \
            shift_quantized_conv2d_bias(quantized_bias, quantized_weight,
                                        input_zero_point)

        quantized_conv = QuantizedConv2d(
            quantized_weight, shifted_quantized_bias,
            input_zero_point, output_zero_point,
            input_scale, weight_scale, output_scale,
            conv.stride, conv.padding, conv.dilation, conv.groups,
            feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
        )

        quantized_backbone.append(quantized_conv)
        ptr += 2
    elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
        quantized_backbone.append(QuantizedMaxPool2d(
            kernel_size=quantized_model.backbone[ptr].kernel_size,
            stride=quantized_model.backbone[ptr].stride
            ))
        ptr += 1
    elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
        quantized_backbone.append(QuantizedAvgPool2d(
            kernel_size=quantized_model.backbone[ptr].kernel_size,
            stride=quantized_model.backbone[ptr].stride
            ))
        ptr += 1
    else:
        raise NotImplementedError(type(quantized_model.backbone[ptr]))  # should not happen
quantized_model.backbone = nn.Sequential(*quantized_backbone)

# finally, quantized the classifier
fc_name = 'classifier'
fc = model.classifier
input_scale, input_zero_point = \
    get_quantization_scale_and_zero_point(
        input_activation[fc_name], feature_bitwidth)

output_scale, output_zero_point = \
    get_quantization_scale_and_zero_point(
        output_activation[fc_name], feature_bitwidth)

quantized_weight, weight_scale, weight_zero_point = \
    linear_quantize_weight_per_channel(fc.weight.data, weight_bitwidth)
quantized_bias, bias_scale, bias_zero_point = \
    linear_quantize_bias_per_output_channel(
        fc.bias.data, weight_scale, input_scale)
shifted_quantized_bias = \
    shift_quantized_linear_bias(quantized_bias, quantized_weight,
                                input_zero_point)

quantized_model.classifier = QuantizedLinear(
    quantized_weight, shifted_quantized_bias,
    input_zero_point, output_zero_point,
    input_scale, weight_scale, output_scale,
    feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
)

def extra_preprocess(x):
    # 원본 FP32 입력 값 범위 (0, 1)를 정수형 범위 (-128, 127)의 int8 형식으로 변환하여 입력합니다.
    return (x * 255 - 128).clamp(-128, 127).to(torch.int8)
per_channel_int8_model_accuracy = evaluate(quantized_model, dataloader['test'],
                               extra_preprocess=[extra_preprocess])

print(f"Per-tensor quantized int8 model has accuracy={per_tensor_int8_model_accuracy:.2f}%")
print(f"Per-channel quantized int8 model has accuracy={per_channel_int8_model_accuracy:.2f}%")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 전체 모델에 대한 K-means quantization

원본 Cell `091`. 이 셀은 **전체 모델에 대한 K-means quantization** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
from torch.nn import parameter
class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name])

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — 2.3. Quantization with PyTorch API

원본 Cell `100`. 이 셀은 **2.3. Quantization with PyTorch API** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# 모델 정의
import torch.nn.functional as F
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()

        # 양자화 스텁 추가
        self.quant = torch.quantization.QuantStub()  # 입력을 양자화
        self.dequant = torch.quantization.DeQuantStub()  # 출력을 역양자화

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Pooling and Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.quant(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.contiguous().view(-1, 128 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        x = self.dequant(x)
        return x
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## Practice 03. Knowledge Distillation 코드 학습

원본: `On-Device AI 강의자료/실습/3. Knowledge Distillation.ipynb`
실습본: `practice_notebooks/on_device_ai/03-knowledge-distillation.ipynb`

### Drill 1 — Define Teacher and Student Models

원본 Cell `010`. 이 셀은 **Define Teacher and Student Models** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class VGGCifar9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, 3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(True)),
            ('conv1', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2)),
            ('conv4', nn.Conv2d(256, 512, 3, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(512)),
            ('relu4', nn.ReLU(True)),
            ('conv5', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn5', nn.BatchNorm2d(512)),
            ('relu5', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv6', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('relu6', nn.ReLU(True)),
            ('conv7', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn7', nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(2)),
        ]))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

class VGGCifar5(nn.Module):
    def __init__(self) -> None:
        # Generate the same scratch model
        set_seed()
        super().__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, 3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(2)),
            ('conv1', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(2)),
        ]))
        self.classifier = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — KD 핵심 개념

원본 Cell `026`. 이 셀은 **KD 핵심 개념** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def train_knowledge_distillation(teacher,
                                 student,
                                 train_loader,
                                 epochs,
                                 learning_rate,
                                 T,  # temperature
                                 soft_target_loss_weight,
                                 ce_loss_weight):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            ##################### YOUR CODE STARTS HERE #####################
            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits =

            # Forward pass with the student model
            student_logits =

            # Soften the student logits by applying softmax
            # Hint: nn.functional.softmax()
            soft_targets =
            student_prob =

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss =

            # Calculate the true label loss
            label_loss =

            # Weighted sum of the two losses
            loss =
            ##################### YOUR CODE ENDS HERE #######################

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — Cosine Similarity 기반 KD 모델 정의

원본 Cell `033`. 이 셀은 **Cosine Similarity 기반 KD 모델 정의** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class VGGCifar9_Cosine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, 3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(True)),
            ('conv1', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2)),
            ('conv4', nn.Conv2d(256, 512, 3, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(512)),
            ('relu4', nn.ReLU(True)),
            ('conv5', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn5', nn.BatchNorm2d(512)),
            ('relu5', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv6', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('relu6', nn.ReLU(True)),
            ('conv7', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn7', nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(2)),
        ]))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        conv_output = torch.flatten(x, 1)
        conv_output_after_pooling = torch.nn.functional.avg_pool1d(conv_output, 2)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x, conv_output_after_pooling

class VGGCifar5_Cosine(nn.Module):
    def __init__(self) -> None:
        # Generate the same scratch model
        set_seed()
        super().__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, 3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(2)),
            ('conv1', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(2)),
        ]))
        self.classifier = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        conv_output = torch.flatten(x, 1)
        x = x.mean([2, 3])
        # conv_output = x
        x = self.classifier(x)
        return x, conv_output
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — [실습 2] Cosine Similarity 기반 KD 학습 함수 정의

원본 Cell `039`. 이 셀은 **[실습 2] Cosine Similarity 기반 KD 학습 함수 정의** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def train_cosine_loss(teacher,
                      student,
                      train_loader,
                      epochs,
                      learning_rate,
                      hidden_rep_loss_weight,
                      ce_loss_weight):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            ##################### YOUR CODE STARTS HERE #####################
            # Forward pass with the teacher model and keep only the hidden representation
            with torch.no_grad():
                _, teacher_hidden_representation =

            # Forward pass with the student model
            student_logits, student_hidden_representation =

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is
            # the case where loss minimization leads to cosine similarity increase.
            # Hint: cosine_loss(x, y, target)에서 target은 1로 이루어진 vector이며, torch.ones(inputs.size(0)).cuda())를 사용
            hidden_rep_loss =

            # Calculate the true label loss
            label_loss =

            # Weighted sum of the two losses
            loss =
            ##################### YOUR CODE ENDS HERE #######################

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — Hint-based KD를 위한 Regressor 포함 모델 정의

원본 Cell `048`. 이 셀은 **Hint-based KD를 위한 Regressor 포함 모델 정의** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class VGGCifar9_Regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, 3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(True)),
            ('conv1', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2)),
            ('conv4', nn.Conv2d(256, 512, 3, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(512)),
            ('relu4', nn.ReLU(True)),
            ('conv5', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn5', nn.BatchNorm2d(512)),
            ('relu5', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv6', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('relu6', nn.ReLU(True)),
            ('conv7', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn7', nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(2)),
        ]))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        conv_feature_map = x
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x, conv_feature_map

class VGGCifar5_Regressor(nn.Module):
    def __init__(self) -> None:
        # Generate the same scratch model
        set_seed()
        super().__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, 3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(True)),
            ('pool0', nn.MaxPool2d(2)),
            ('conv1', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(2)),
        ]))
        self.regressor = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        regressor_output = self.regressor(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x, regressor_output
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## Practice 04. Pruning for LLM 코드 학습

원본: `On-Device AI 강의자료/실습/4. Pruning for LLM.ipynb`
실습본: `practice_notebooks/on_device_ai/04-pruning-llm.ipynb`

### Drill 1 — 모델 평가

원본 Cell `006`. 이 셀은 **모델 평가** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def evaluate(model, tokenizer):
    """
    모델의 perplexity를 계산하는 함수입니다.

    Args:
        model: 평가할 모델
        tokenizer: 토크나이저

    Returns:
        float: 계산된 perplexity 값
    """
    # 테스트 데이터셋 로드 및 전처리
    testenc = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    # 입력 데이터를 모델 디바이스로 이동
    testenc = testenc.input_ids.to(model.device)
    nsamples = 40
    model = model.eval()  # 평가 모드로 설정

    # Negative log likelihood 계산
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        # 배치 데이터 준비
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)

        # 모델 추론
        with torch.no_grad():
            lm_logits = model(batch).logits

        # 로짓과 레이블 시프트
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]

        # 손실 계산
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    # Perplexity 계산 및 반환
    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — [실습 1] Magnitude-based Pruning 구현

원본 Cell `010`. 이 셀은 **[실습 1] Magnitude-based Pruning 구현** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
@torch.no_grad()
def prune_magnitude(model, sparsity):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            ##################### YOUR CODE STARTS HERE #####################
            num_elements =
            num_zeros =
            importance =
            threshold =
            mask =
            ##################### YOUR CODE ENDS HERE #######################
            W.mul_(mask)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — [실습 2] Calibration 데이터셋 준비

원본 Cell `016`. 이 셀은 **[실습 2] Calibration 데이터셋 준비** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
@torch.no_grad()
def get_calib_feat(model, tokenizer, samples):
    input_dict = dict()
    nsamples_dict = dict()
    def add_batch(m, x, y, name):
        if name not in input_dict:
            input_dict[name] = torch.zeros((m.weight.data.shape[1]), device=m.weight.data.device)
            nsamples_dict[name] = 0

        if isinstance(x, tuple):
            x = x[0]

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        tmp = x.shape[0]
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        x = x.t()

        input_dict[name] *= nsamples_dict[name] / (nsamples_dict[name] + tmp)
        nsamples_dict[name] += tmp

        x = x.type(torch.float32)
        ##################### YOUR CODE STARTS HERE #####################
        # activation_norm을 계산하세요.
        # x.shape => (hidden_size, batch_size)
        activation_norm =
        # activation_norm.shape => (hidden_size)
        ##################### YOUR CODE ENDS HERE #######################
        input_dict[name] += activation_norm / nsamples_dict[name]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in name:
            hooks.append(
                m.register_forward_hook(
                    partial(add_batch, name=name)))

    print("Collecting norm of input activations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for key in input_dict.keys():
        input_dict[key].sqrt_()

    for hook in hooks:
        hook.remove()
    return input_dict
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — [실습 3] Wanda Pruning 구현

원본 Cell `019`. 이 셀은 **[실습 3] Wanda Pruning 구현** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
@torch.no_grad()
def prune_wanda(model, sparsity, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            ##################### YOUR CODE STARTS HERE #####################
            row, col =
            num_zeros_per_row =
            importance =
            threshold =
            mask =
            ##################### YOUR CODE ENDS HERE #######################
            W.mul_(mask)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## Practice 05. Quantization for LLM 코드 학습

원본: `On-Device AI 강의자료/실습/5. Quantization for LLM.ipynb`
실습본: `practice_notebooks/on_device_ai/05-quantization-llm.ipynb`

### Drill 1 — Setup

원본 Cell `009`. 이 셀은 **Setup** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class LLMModel:
    """
    대형언어모델(LLM)의 평가 및 관리 클래스.
    """

    def __init__(self, model_name, weights_path=None):
        self.model_name = model_name
        self.weights_path = weights_path

        if (self.weights_path != None):
            self._load_model_from_weights()
        else:
            # 사전학습된 언어모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            self.model.eval()  # 평가 모드 설정 (dropout 비활성화)

        # 토크나이저 로드 (토크나이저는 아키텍처와 세트이므로 from_pretrained 유지)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # 평가용 데이터셋 로드 (WikiText-2)
        testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = self.tokenizer("\n\n".join(testenc['text']), return_tensors='pt')
        self.testenc = testenc.input_ids.to(self.model.device)

        self.model_changed = False

    def _load_model_from_weights(self):
        print(f"Loading model architecture: {self.model_name}")
        config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_config(config)

        print(f"Loading weights from: {self.weights_path}")
        checkpoint = torch.load(self.weights_path, map_location="cpu")

        # --- [수정] state_dict 키 확인 및 추출 로직 추가 ---
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print("Error loading state_dict directly. Attempting strict=False...")
            print(e)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(dtype=torch.float16)
        self.model.cuda()
        self.model.eval()

    def _evaluate(self):
        nsamples = 100
        nlls = []

        for i in tqdm(range(nsamples), desc="evaluating..."):
            batch = self.testenc[:, (i * 2048):((i + 1) * 2048)].to(self.model.device)
            with torch.no_grad():
                lm_logits = self.model(batch).logits

            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

    def get_model_size(self, data_width=16, group_size=-1):
        if group_size != -1:
            data_width += (16 + 4) / group_size
        num_elements = sum(param.numel() for param in self.model.parameters())
        return num_elements * data_width

    def model_delete(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def model_evaluate(self, data_width, group_size):
        model_perplexity = self._evaluate()
        model_size = self.get_model_size(data_width=data_width, group_size=group_size)

        print(f"\nmodel perplexity: {model_perplexity:.2f}")
        print(f"model size: {model_size / 1024 / 1024 / 8:.2f} MiB")
        return model_perplexity

    def model_reset(self):
        """
        모델이 변경된 경우 초기 가중치 파일로 재설정
        """
        if self.model_changed:
            self.model_delete()
            # 초기 로드 로직 재사용
            if (self.weights_path != None):
                self._load_model_from_weights()
            else:
                # 사전학습된 언어모델 로드
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                self.model.eval()  # 평가 모드 설정 (dropout 비활성화)
            self.model_changed = False

    def model_change(self, model: nn.Module):
        self.model_delete()
        self.model = model
        self.model.eval()
        self.model_changed = True
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — Pseudo Quantization

원본 Cell `022`. 이 셀은 **Pseudo Quantization** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)

    samples = []
    n_run = 0

    # n_samples 만큼 문장을 토크나이즈하여 샘플 생성
    for data in dataset:
        line = data["text"].strip()
        line_encoded = tokenizer.encode(line)

        # 너무 긴 문장은 제외
        if len(line_encoded) > block_size:
            continue

        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue

        samples.append(sample)
        n_run += 1

        # 설정된 개수만큼 샘플 수집 완료 시 종료
        if n_run == n_samples:
            break

    # 모든 샘플을 연결 후 block_size 단위로 나눔
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")

    # block_size 단위로 분할된 텐서 리스트 반환
    return [cat_samples[:, i * block_size:(i + 1) * block_size] for i in range(n_split)]


@torch.no_grad()
def get_calib_feat(model, tokenizer):
    # 각 Linear layer의 입력 활성값(activation) 통계를 수집하기 위한 훅 등록용 dict
    input_dict = dict()

    # forward hook 함수: layer 입력의 평균 절댓값(activation scale)을 계산
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        # 입력 텐서를 (batch*seq_len, hidden_dim) 형태로 reshape 한 뒤,
        # hidden_dim 방향으로 평균 절댓값 계산 → activation scale 추정
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()

        # 레이어별로 통계 누적
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    # 모든 Linear layer에 대해 forward hook 등록
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)
                )
            )

    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # calibration용 입력 데이터셋 생성
    samples = get_calib_dataset(tokenizer)
    pbar = tqdm(samples)

    # 각 샘플을 모델에 통과시키며 hook으로 입력 통계 수집
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    # hook 해제
    for hook in hooks:
        hook.remove()

    # 레이어별 입력 통계(activation scale) 반환
    return input_dict
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — [실습 2] Scale factor search

원본 Cell `034`. 이 셀은 **[실습 2] Scale factor search** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
@torch.no_grad()
def auto_scale_block(module, name, w_bit,
                     q_group_size,
                     input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)
        s_x = torch.clamp(s_x, 1e-5)


        # Step 1: best_error, best_ratio, 및 best_scales를 초기화
        best_error = torch.inf
        best_ratio = -1
        best_scales = 0


        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: 공식에 따라 스케일 계산
            scales =

            ############### YOUR CODE ENDS HERE #################
            assert scales.shape == s_x.shape

            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:

                scales = scales.to(fc.weight.device)

                ############### YOUR CODE STARTS HERE ###############

                # Step 3: scale_factor를 이용해 중요한 가중치 채널의 값을 확대합니다.
                fc.weight.data

                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)

                # Step 4: scale_factor를 이용해 중요한 가중치 채널의 값을 다시 축소하세요.
                fc.weight.data

                ############### YOUR CODE ENDS HERE #################

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # (1) Self-Attention 입력 부분 (Q, K, V projection)
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # (2) Self-Attention 출력 부분 (out_proj)
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.self_attn.out_proj, [module.self_attn.out_proj], inp)
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # (3) Feed-Forward Network 첫 번째 FC (fc1)
    inp = input_feat[name + '.fc1']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # (4) Feed-Forward Network 두 번째 FC (fc2)
    inp = input_feat[name + '.fc2']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    scale_fc_fc(module.fc1, module.fc2, final_scales)

@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(model, w_bit, q_group_size, input_feat):
    """
    OPT 계열 모델 전체에 대해 자동 스케일 조정(auto-scale) 기반 pseudo quantization 수행.
    각 Decoder Layer 단위로 auto_scale_block을 적용.
    """
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    # 각 OPTDecoderLayer(Transformer 블록)에 대해 auto scaling 수행
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    # 모든 Linear layer에 pseudo quantization 최종 적용
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=q_group_size
            )
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — Pseudo Quantization

원본 Cell `042`. 이 셀은 **Pseudo Quantization** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class W8A8Linear(nn.Module):
    """
    8-bit Weight(A) × 8-bit Activation(A) 양자화 시뮬레이션용 Linear Layer.
    실제 연산은 FP16으로 수행되지만, 연산 전후에 quantize/dequantize 단계를 통해
    8-bit 정밀도의 효과를 모사
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        quantize_bits=8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight 초기화 (FP16)
        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )

        # Bias 초기화 (FP16)
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        # 입력 activation quantization 방식 선택
        # per_token: 토큰별로 스케일 계산
        # per_tensor: 전체 텐서 단위로 스케일 계산
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        # 출력 activation quantization 옵션 (예: BMM 입력용)
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x  # 출력 quantization 미적용

        self.quantize_bits = quantize_bits

    def to(self, *args, **kwargs):
        # .to() 호출 시 weight와 bias도 동일한 디바이스로 이동
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        """
        순전파 (forward pass)
        1. 입력 activation quantization
        2. Linear 연산
        3. 출력 quantization
        """
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False, quantize_bits=8
    ):
        """
        FP16 Linear 모듈을 받아, W8A8Linear로 변환
        Weight를 8-bit로 quantize하고, activation quantization 함수를 연결함.
        """
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )

        # Weight quantization 방식 선택
        if weight_quant == "per_channel":
            # 출력 채널별 스케일 계산
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=new_module.quantize_bits
            )
        elif weight_quant == "per_tensor":
            # 전체 텐서 단위 스케일 계산
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=new_module.quantize_bits
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")

        new_module.weight_quant_name = weight_quant

        # bias 복사
        if module.bias is not None:
            new_module.bias = module.bias

        return new_module

    def __repr__(self):
        return (
            f"W8A8Linear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, "
            f"output_quant={self.output_quant_name})"
        )
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 5 — Pseudo Quantization

원본 Cell `043`. 이 셀은 **Pseudo Quantization** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    """
    weight per-channel quantization (출력 채널별 최대값 스케일)
    """
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    """
    weight per-tensor quantization (전체 텐서의 absmax 기준)
    """
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    """
    activation per-token quantization:
    각 토큰(=시퀀스 단위)마다 absmax 기준으로 스케일링.
    """
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    """
    activation per-tensor quantization:
    전체 텐서의 absmax를 기준으로 스케일링.
    """
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

def quantize_opt(
    model,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=True,
    quantize_bits=8,
):
    """
    OPT 모델에 W8A8 quantization을 적용하는 함수.
    Linear 및 Attention projection 계층을 W8A8Linear로 교체함.

    - weight_quant: 'per_channel' 또는 'per_tensor'
    - act_quant: 'per_token' 또는 'per_tensor'
    - quantize_bmm_input: True이면 q_proj, k_proj, v_proj 출력도 quantize하여 BMM 입력 시뮬레이션
    """
    from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            # Feed-Forward 계층 (fc1, fc2) quantization
            m.fc1 = W8A8Linear.from_float(
                m.fc1,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bits=quantize_bits,
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bits=quantize_bits,
            )

        elif isinstance(m, OPTAttention):
            # Attention 내부 q, k, v, out projection quantization
            # quantize_bmm_input=True이면 BMM 입력용으로 q/k/v 출력도 quantize 시뮬레이션
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quantize_bits=quantize_bits,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quantize_bits=quantize_bits,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                quantize_bits=quantize_bits,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bits=quantize_bits,
            )

    return model
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?
