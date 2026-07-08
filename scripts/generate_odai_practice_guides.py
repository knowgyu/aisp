from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / 'On-Device AI 강의자료' / '실습'
OUT_DIR = ROOT / 'study_notes' / 'on_device_ai' / 'practice'

PRACTICES = [
    {
        'idx': '01',
        'title': 'Pruning for CNN',
        'notebook': '1. Pruning for CNN.ipynb',
        'out': '01_pruning_cnn_practice_guide.md',
        'lecture': 'ODAI-1 Chapter 2 Network Pruning',
        'goal': 'VGG/CIFAR-10 모델에서 pruning granularity, pruning ratio, sensitivity, pruning schedule을 코드로 연결한다.',
        'flow': ['Dense VGG', 'Weight 분포/기준 측정', 'Granularity별 mask', 'Sensitivity scan', 'Layer-wise vs global', 'Schedule + fine-tuning'],
        'global_shapes': [
            ('CIFAR-10 batch', '`[B, 3, 32, 32]`', 'RGB 이미지 mini-batch'),
            ('Conv weight', '`[C_out, C_in, K_h, K_w]`', 'pruning granularity가 정의되는 기본 텐서'),
            ('mask', '`same as weight`', '`1`은 유지, `0`은 제거'),
            ('logits', '`[B, 10]`', 'CIFAR-10 class score'),
        ],
        'formula': r'$W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$'
    },
    {
        'idx': '02',
        'title': 'Quantization for CNN',
        'notebook': '2. Quantization for CNN.ipynb',
        'out': '02_quantization_cnn_practice_guide.md',
        'lecture': 'ODAI-1 Chapter 3 Quantization',
        'goal': 'VGG/CIFAR-10에서 uniform quantization, integer-only inference, k-means/non-uniform quantization, PTQ/QAT 흐름을 구현 관점으로 익힌다.',
        'flow': ['FP32 baseline', 'bit range/scale/zero-point', 'weight/activation quantize', 'quantized layer', 'integer-only inference', 'PTQ/QAT 비교'],
        'global_shapes': [
            ('FP32 weight', '`float32 tensor`', '학습된 연속값 파라미터'),
            ('quantized weight', '`int8/int4 tensor + scale/zero_point`', '저비트 정수 표현'),
            ('activation', '`[B, C, H, W]`', 'calibration/observer가 range를 추정하는 대상'),
            ('accumulator', '`int32`', '정수 GEMM/Conv의 누산 타입'),
        ],
        'formula': r'$q=\mathrm{clip}(\mathrm{round}(x/s)+z,q_{min},q_{max}),\quad \hat{x}=s(q-z)$'
    },
    {
        'idx': '03',
        'title': 'Knowledge Distillation',
        'notebook': '3. Knowledge Distillation.ipynb',
        'out': '03_knowledge_distillation_practice_guide.md',
        'lecture': 'ODAI-1 Chapter 4 Knowledge Distillation',
        'goal': '큰 teacher의 logits/feature 표현을 작은 student에 전달하는 KD 손실을 직접 구현하고 CE 학습과 비교한다.',
        'flow': ['Teacher/Student 준비', 'CE baseline', 'Logit KD', 'Temperature/KL', 'Feature KD', 'Hint/regressor KD'],
        'global_shapes': [
            ('image batch', '`[B, 3, 32, 32]`', 'CIFAR-10 입력'),
            ('logits', '`[B, 10]`', 'teacher/student class score'),
            ('soft targets', '`softmax(logits/T)`', 'class 간 유사도 정보'),
            ('feature map', '`[B, C, H, W]`', 'representation-level KD 대상'),
        ],
        'formula': r'$\mathcal{L}=\alpha T^2 KL(p_T^t\Vert p_T^s)+(1-\alpha)CE(y,p^s)$'
    },
    {
        'idx': '04',
        'title': 'Pruning for LLM',
        'notebook': '4. Pruning for LLM.ipynb',
        'out': '04_pruning_llm_practice_guide.md',
        'lecture': 'ODAI-2 Chapter 1 LLM Pruning',
        'goal': 'SmolLM 계열 causal LM에서 perplexity 평가, magnitude pruning, Wanda activation-aware pruning, N:M structured pruning을 연결한다.',
        'flow': ['Load LM/tokenizer', 'Wikitext perplexity', 'Magnitude pruning', 'Calibration activations', 'Wanda score', 'N:M sparsity'],
        'global_shapes': [
            ('token ids', '`[B, T]`', '언어 모델 입력 토큰'),
            ('hidden states', '`[B, T, d_model]`', 'linear layer 입력 activation'),
            ('linear weight', '`[d_out, d_in]`', 'LLM pruning 주요 대상'),
            ('PPL', '`exp(cross_entropy)`', 'pruning 후 품질 지표'),
        ],
        'formula': r'$score_{ij}=|W_{ij}|\cdot \|X_j\|_2$  (Wanda의 핵심 직관)'
    },
    {
        'idx': '05',
        'title': 'Quantization for LLM',
        'notebook': '5. Quantization for LLM.ipynb',
        'out': '05_quantization_llm_practice_guide.md',
        'lecture': 'ODAI-2 Chapter 2 LLM Quantization',
        'goal': 'OPT/TinyLlama에서 weight-only quantization, AWQ, W8A8/SmoothQuant, rotation 기반 quantization을 구현 흐름으로 정리한다.',
        'flow': ['Baseline PPL/size', 'Pseudo W-only quant', 'AWQ calibration', 'Auto scale', 'W8A8 SmoothQuant', 'Rotation/LayerNorm fusion'],
        'global_shapes': [
            ('weight matrix', '`[out_features, in_features]`', 'group-wise quantization 대상'),
            ('group', '`q_group_size` columns', 'scale/zero-point를 공유하는 단위'),
            ('activation outlier', '`channel-wise max`', 'AWQ/SmoothQuant가 다루는 핵심 문제'),
            ('rotation matrix', '`[d_model, d_model]`', 'outlier를 섞어 양자화 난이도를 낮추는 직교행렬'),
        ],
        'formula': r'$Q(w)=s\cdot(\mathrm{round}(w/s)-z),\quad s_q=(\alpha-\beta)/(2^b-1)$'
    },
]


def clean_text(s: str, n: int = 320) -> str:
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'!\[[^\]]*\]\([^)]*\)', '그림', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:n] + ('…' if len(s) > n else '')


def first_heading(src: str) -> str | None:
    for line in src.splitlines():
        m = re.match(r'\s{0,3}(#{1,6})\s+(.+)', line)
        if m:
            return clean_text(m.group(2), 120)
    return None


def code_signature(src: str) -> str:
    lines = [ln.rstrip() for ln in src.splitlines() if ln.strip()]
    sigs = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(('def ', 'class ', '@')) or re.match(r'[A-Za-z_][\w\.]*\s*=\s*', stripped):
            sigs.append(stripped)
        elif stripped.startswith(('for ', 'with ', 'if ', 'print(', 'plt.', 'model.', 'llm_model.')):
            sigs.append(stripped)
        if len(sigs) >= 3:
            break
    return (' / '.join(sigs)[:220]).rstrip() if sigs else clean_text(src, 220)


def ast_symbols(src: str) -> tuple[list[str], list[str]]:
    defs, classes = [], []
    try:
        tree = ast.parse(src)
    except Exception:
        return defs, classes
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defs.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return defs, classes


def detect_topics(text: str) -> list[str]:
    lower = text.lower()
    rules = [
        ('seed', '재현성'), ('cifar', 'CIFAR-10 데이터'), ('dataloader', 'DataLoader'),
        ('vgg', 'VGG 모델'), ('evaluate', '평가 루프'), ('train(', '학습 루프'),
        ('parameter', '모델 크기'), ('sparsity', '희소도'), ('prun', '프루닝'),
        ('mask', '마스크'), ('fine', 'fine-grained'), ('vector', 'vector-level'),
        ('kernel', 'kernel-level'), ('channel', 'channel-level'), ('sensitivity', '민감도 분석'),
        ('scheduler', '스케줄링'), ('quant', '양자화'), ('scale', 'scale'), ('zero', 'zero-point'),
        ('observer', 'observer/calibration'), ('calib', 'calibration'), ('qconfig', 'PyTorch quant config'),
        ('kmeans', 'k-means/non-uniform'), ('qat', 'QAT'), ('ptq', 'PTQ'),
        ('teacher', 'teacher'), ('student', 'student'), ('distill', 'distillation'), ('temperature', 'temperature'),
        ('kl', 'KL divergence'), ('cosine', 'cosine feature loss'), ('mse', 'MSE feature loss'),
        ('tokenizer', 'tokenizer'), ('perplexity', 'perplexity'), ('wikitext', 'Wikitext'), ('wanda', 'Wanda'),
        ('activation', 'activation'), ('awq', 'AWQ'), ('smooth', 'SmoothQuant'), ('rotation', 'rotation'),
        ('layernorm', 'LayerNorm'), ('linear', 'Linear layer'), ('opt', 'OPT'), ('tinyllama', 'TinyLlama'),
    ]
    out = []
    for key, label in rules:
        if key in lower and label not in out:
            out.append(label)
    return out[:6]


def explanation_for(config: dict, cell_type: str, src: str, idx: int) -> list[str]:
    text = src.lower()
    title = config['title']
    bullets = []
    if cell_type == 'markdown':
        bullets.append('이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.')
    else:
        bullets.append('이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.')
    # shared rules
    if 'seed' in text or 'manual_seed' in text:
        bullets.append('난수 seed를 고정한다. pruning threshold, data augmentation, optimizer 순서가 바뀌면 비교가 흔들리므로 baseline과 실험 조건을 고정하는 역할이다.')
    if 'dataloader' in text or 'cifar' in text or 'load_dataset' in text:
        bullets.append('데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.')
    if 'evaluate' in text or 'perplexity' in text or 'accuracy' in text:
        bullets.append('모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.')
    if 'train' in text or 'optimizer' in text or 'scheduler' in text:
        bullets.append('학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.')
    if 'prun' in text or 'sparsity' in text or 'mask' in text:
        bullets.append('pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.')
    if 'fine' in text and 'prun' in text:
        bullets.append('fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.')
    if 'vector' in text or 'kernel' in text or 'channel' in text:
        bullets.append('structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.')
    if 'sensitivity' in text:
        bullets.append('sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.')
    if 'quant' in text or 'scale' in text or 'zero' in text:
        bullets.append('quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.')
    if 'observer' in text or 'calib' in text:
        bullets.append('calibration/observer는 activation range를 샘플 데이터로 추정한다. range가 좁으면 clipping, 넓으면 resolution 손실이 생긴다.')
    if 'kmeans' in text:
        bullets.append('k-means/non-uniform quantization은 균일 간격 대신 centroid codebook으로 값을 대표한다. 분포가 0 근처에 몰린 weight에서 오차를 줄일 수 있지만 하드웨어 구현은 더 복잡하다.')
    if 'teacher' in text or 'student' in text or 'distill' in text:
        bullets.append('KD에서는 teacher가 만든 부드러운 확률분포나 feature를 student의 학습 신호로 쓴다. label 하나만 보는 CE보다 class 간 관계를 더 많이 전달한다.')
    if 'temperature' in text or 'kl' in text:
        bullets.append('temperature `T`가 커지면 softmax가 완만해져 “2등/3등 class가 얼마나 그럴듯한지”가 드러난다. KL 손실에는 보통 `T^2` 보정이 붙는다.')
    if 'cosine' in text or 'mse' in text or 'regressor' in text or 'feature' in text:
        bullets.append('feature-level KD는 logits뿐 아니라 중간 representation을 맞춘다. teacher/student 채널 수가 다르면 regressor로 shape를 맞춘 뒤 MSE/Cosine 손실을 적용한다.')
    if 'wanda' in text:
        bullets.append('Wanda는 weight magnitude만 보지 않고 입력 activation 크기도 곱한다. 같은 weight라도 자주/크게 활성화되는 channel에 연결되면 더 중요하다고 보는 방식이다.')
    if 'awq' in text or 'outlier' in text:
        bullets.append('AWQ는 LLM activation outlier 때문에 단순 weight quantization이 깨지는 문제를 다룬다. 중요한 channel을 scale로 보호해 weight-only 저비트 양자화 오차를 줄인다.')
    if 'smooth' in text:
        bullets.append('SmoothQuant는 activation의 어려움을 weight 쪽으로 일부 이전한다. 수식상 equivalent transformation이라 FP 결과는 유지하되 W8A8 quantization이 쉬워진다.')
    if 'rotation' in text or 'quarot' in text or 'spinquant' in text:
        bullets.append('rotation 기반 방법은 직교행렬로 hidden dimension을 섞어 outlier를 완화한다. 직교변환은 정보량을 보존하지만 quantization grid에 더 잘 맞는 분포를 만들 수 있다.')
    if not bullets[1:]:
        topics = detect_topics(src)
        if topics:
            bullets.append('핵심 키워드: ' + ', '.join(topics) + '. 이 셀에서 해당 키워드가 코드 변수/함수로 어떻게 구현되는지 오른쪽 노트북에서 확인한다.')
        else:
            bullets.append('주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.')
    return bullets[:5]


def shape_notes(config: dict, src: str) -> list[str]:
    text = src.lower()
    notes = []
    if 'conv' in text or 'cifar' in text or 'vgg' in text:
        notes.append('CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.')
    if 'linear' in text or 'llm' in text or 'opt' in text or 'tinyllama' in text:
        notes.append('Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.')
    if 'mask' in text or 'prun' in text:
        notes.append('mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.')
    if 'quant' in text or 'scale' in text or 'zero' in text:
        notes.append('quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.')
    if 'softmax' in text or 'kl' in text or 'temperature' in text:
        notes.append('logits `[B,num_classes]`에 softmax를 적용하면 class 확률분포가 된다. temperature는 softmax 전 logits를 `T`로 나누는 조절값이다.')
    if 'perplexity' in text or 'cross_entropy' in text:
        notes.append('PPL은 `exp(average cross entropy)`다. 낮을수록 다음 token 예측이 좋다.')
    if not notes:
        notes.append(config['formula'])
    return notes[:3]


def implementation_checks(src: str) -> list[str]:
    text = src.lower()
    checks = []
    if 'your code' in text or 'todo' in text:
        checks.append('`YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.')
    if 'cuda' in text or '.to(' in text:
        checks.append('입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.')
    if 'clone' in text or 'deepcopy' in text or 'recover' in text or 'reset' in text:
        checks.append('비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.')
    if 'no_grad' in text or 'inference_mode' in text:
        checks.append('평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.')
    if 'round' in text or 'clamp' in text:
        checks.append('round 후 clamp 순서가 중요하다. 정수 범위를 넘으면 overflow/잘못된 dequantization이 생긴다.')
    if 'view' in text or 'reshape' in text or 'transpose' in text:
        checks.append('reshape/transpose 후 어떤 축이 channel/group/kernel 축인지 반드시 주석으로 적어보는 것이 좋다.')
    if not checks:
        checks.append('이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.')
    return checks[:3]


def section_rows(nb):
    headings = []
    for i, c in enumerate(nb['cells'], 1):
        if c['cell_type'] == 'markdown':
            h = first_heading(''.join(c.get('source','')))
            if h:
                headings.append((i, h))
    rows = []
    for j, (start, h) in enumerate(headings):
        end = headings[j+1][0]-1 if j+1 < len(headings) else len(nb['cells'])
        rows.append((start, end, h))
    return rows


def write_guide(config: dict):
    nb = json.loads((NB_DIR / config['notebook']).read_text(encoding='utf-8'))
    cells = nb['cells']
    out = []
    out.append(f"# On-Device AI Practice {config['idx']} — {config['title']} 셀별 코드 학습 가이드\n")
    out.append(f"> 오른쪽에는 원본 노트북 HTML을 띄우고, 왼쪽은 **모든 셀을 따라가는 해설 지도**로 쓴다. 이 문서는 원본 코드를 대체하지 않고, 각 셀이 왜 필요한지/무슨 shape로 움직이는지/직접 구현할 때 어디를 봐야 하는지 설명한다.\n")
    out.append(f"- 기준 교안: `{config['lecture']}`")
    out.append(f"- 원본 노트북: `On-Device AI 강의자료/실습/{config['notebook']}`")
    out.append(f"- 학습 목표: {config['goal']}\n")
    out.append('## 0. 그림으로 먼저 잡기\n')
    nodes = [chr(ord('A')+i) for i in range(len(config['flow']))]
    flow_lines = ['flowchart LR']
    for i, label in enumerate(config['flow']):
        flow_lines.append(f'  {nodes[i]}["{label}"]')
        if i:
            flow_lines.append(f'  {nodes[i-1]} --> {nodes[i]}')
    out.append('```mermaid\n' + '\n'.join(flow_lines) + '\n```\n')
    out.append('### 실습 전체에서 계속 붙잡을 수식\n')
    out.append(config['formula'] + '\n')
    out.append('### 핵심 shape 표\n')
    out.append('| 대상 | Shape / 표현 | 의미 |')
    out.append('|---|---|---|')
    for name, shape, meaning in config['global_shapes']:
        out.append(f'| {name} | {shape} | {meaning} |')
    out.append('')
    out.append('## 1. 노트북 전체 지도\n')
    out.append('| 구간 | 셀 범위 | 오른쪽에서 보이는 제목 | 여기서 잡아야 할 것 |')
    out.append('|---:|---:|---|---|')
    rows = section_rows(nb)
    for k, (s, e, h) in enumerate(rows, 1):
        # choose a topic from cells in range
        merged = ' '.join(''.join(c.get('source','')) for c in cells[s-1:e])
        topics = ', '.join(detect_topics(merged)[:4]) or '변수 흐름, shape, 평가값'
        out.append(f'| {k} | {s:03d}-{e:03d} | {h} | {topics} |')
    out.append('')
    out.append('## 2. 셀별 Walkthrough\n')
    out.append('아래 번호는 오른쪽 노트북의 cell 순서와 맞춘 것이다. Markdown 셀도 건너뛰지 않는다. Markdown 셀은 바로 다음 코드가 어떤 문제를 푸는지 정의하는 경우가 많기 때문이다.\n')
    current_section = '도입'
    for i, c in enumerate(cells, 1):
        src = ''.join(c.get('source','')).strip()
        if not src:
            continue
        h = first_heading(src)
        if c['cell_type'] == 'markdown' and h:
            current_section = h
        ctype = 'Markdown' if c['cell_type'] == 'markdown' else 'Code'
        title = h or code_signature(src)
        out.append(f"### Cell {i:03d} · {ctype} · {title}\n")
        out.append(f"- **현재 구간**: {current_section}")
        if c['cell_type'] == 'markdown':
            out.append(f"- **오른쪽에서 읽을 내용**: {clean_text(src, 360)}")
        else:
            defs, classes = ast_symbols(src)
            sig = code_signature(src)
            out.append(f"- **오른쪽에서 볼 코드**: `{sig.replace('`','')}`")
            if classes:
                out.append(f"- **정의되는 class**: `{', '.join(classes)}`")
            if defs:
                out.append(f"- **정의되는 함수**: `{', '.join(defs)}`")
        out.append('- **무슨 작업인가**:')
        for b in explanation_for(config, c['cell_type'], src, i):
            out.append(f'  - {b}')
        out.append('- **수학/shape 관점**:')
        for b in shape_notes(config, src):
            out.append(f'  - {b}')
        out.append('- **직접 구현할 때 체크**:')
        for b in implementation_checks(src):
            out.append(f'  - {b}')
        if c['cell_type'] == 'code':
            topics = detect_topics(src)
            if topics:
                out.append(f"- **키워드**: {', '.join(topics)}")
        out.append('')
    out.append('## 3. 실습 후 스스로 확인할 질문\n')
    out.extend([
        '1. 이 노트북에서 baseline metric은 무엇이고, 압축 후 얼마만큼 변했는가?',
        '2. in-place로 model weight를 바꾸는 셀은 어디이며, 원본 복구/reset은 어떻게 하는가?',
        '3. 핵심 함수 하나를 빈 파일에 다시 구현한다면 입력/출력 shape를 주석으로 쓸 수 있는가?',
        '4. 정확도/PPL 손실이 생겼을 때 원인이 range 추정, mask 단위, scale 선택, calibration 부족 중 어디에 가까운가?',
        '5. 실제 hardware speedup으로 이어지려면 단순 parameter 감소 외에 어떤 조건이 필요한가?',
        ''
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / config['out']).write_text('\n'.join(out), encoding='utf-8')
    print(config['out'], len(cells), 'cells ->', len(out), 'lines')

def main() -> None:
    if '--force-generated' not in sys.argv:
        print(
            'Refusing to overwrite curated ODAI practice guides. '
            'These files are now manually maintained for study quality. '
            'Pass --force-generated only if you intentionally want to regenerate heuristic drafts.',
            file=sys.stderr,
        )
        raise SystemExit(2)
    for cfg in PRACTICES:
        write_guide(cfg)


if __name__ == '__main__':
    main()
