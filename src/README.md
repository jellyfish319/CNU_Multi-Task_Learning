# 실험 계획

## 1. 데이터셋 다운로드

ImageNet: [https://image-net.org/request](https://image-net.org/request), 로그인 후 request 받아야함
COCO2017 train, val: [https://cocodataset.org/#download](https://cocodataset.org/#download), 다운로드가 안될 경우 시크릿창\(Chrome\)에서 시도
ADE20K: [https://ade20k.csail.mit.edu/request_data/index.php](https://ade20k.csail.mit.edu/request_data/index.php), 로그인 후 다운

## 2. 모형 설계

Backbone Network 모형은 PVT v2를 기반으로 설계 후 Multi-Task Learning을 적용시키는 것이 목적
MTL 구현 관련 참고 깃헙
LibMTL: https://github.com/median-research-group/LibMTL
PVT v2: https://huggingface.co/papers/2106.13797
