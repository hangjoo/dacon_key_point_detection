# 모션 키포인트 검출 AI 경진대회

기간 : 2021.02.10 ~ 2021.04.05

**Public 4위 34.16211점**

**Private 6위 7.36127점**

특정 운동 동작을 수행하고 있는 사람의 지정된 신체 부위의 키포인트를 검출하는 대회입니다. 

FAIR(Facebook AI Research)에서 만든 Detectron2를 사용하여 사전학습된 Keypoint RCNN 모델을 사용했으며, 같은 모델을 다양한 augmentation으로 transform하여 학습된 모델의 결과를 앙상블하여 해당 스코어를 기록했습니다.

성능을 높이기 위해 augmentation transform을 바꿔가면서 실험 했습니다. 회전과 크롭을 포함한 augmentation을 사용했을 때 성능 향상을 보였고, 이미지에 노이즈를 주거나 하는 augmentation의 경우 뚜렷한 향상이 없거나 오히려 스코어가 감소하는 결과를 보여 적용하지 않았습니다.

train.py에 실험 관리를 위한 neptune api 관련 코드들이 있어 실행시 인증 문제로 오류가 발생합니다. neptune과 관련된 코드들을 주석 처리하고 실행시키시길 바랍니다.



## 데이터

학습 이미지 4195장, 테스트 이미지 1600장이 존재하며 각 이미지에 대응되는 신체의 24개 지점의 키포인트의 x, y 좌표와 파일명을 포함하는 csv 파일로 구성되어 있습니다. 주어진 csv 파일에서 잘못된 키포인트가 존재하여 해당 키포인트를 수정하여 사용했습니다.



## 파일 구조

### code

#### augmentation.py

Albumentations 라이브러리를 사용했으며, pytorch에서 제공하는 transform 등과 비교하여 속도가 빠르다고 합니다. 거기에 키포인트도 함께 변환해주는 기능도 있어 사용했습니다.
transform_dict에서 미리 정의해놓고 아래에서 문자열 값을 리스트에 간단히 추가해서 다양하게 augmentation을 적용해서 모델 성능을 측정했습니다. Augmentation된 이미지들은 따로 폴더를 생성하여 저장해놓고 사용했습니다. Augmentation을 적용할 경로를 original 폴더로 설정하고 original.csv와 images 폴더에서 이미지를 가져와 사용했습니다.

#### Trainer.py

Detectron2에서 사용되는 Trainer를 커스텀하여 사용하기 위해 사용되었습니다. Detectron2은 기본적으로 CFGNode라는 클래스를 사용하여 전반적인 모델, 학습 등의 config들을 저장하여 사용합니다. Detectron2 공식문서(https://detectron2.readthedocs.io/en/latest/modules/config.html?highlight=CFGNode)에서 해당 내용을 읽어보시는 것을 추천드립니다.

#### utils.py

데이터셋을 만들 때 사용되거나 결과를 저장하는 등에 필요해 구현된 함수들을 한 곳에 모아놓고 import 해서 사용했습니다.

**train_val_split**: 학습 데이터셋과 검증 데이터 셋을 나눌 때 사용되며 augmentation된 이미지와 original 이미지들이 섞여있기 때문에 검증 데이터셋에 original 이미지만 포함되도록 구현했습니다.

**get_data_dicts**: Detectron2에서 데이터셋을 생성할 때 사용되는 함수입니다. Detectron2의 경우 데이터를 딕셔너리 형태로 제공받아 사용하기 때문에 해당 타입에 맞게 알맞은 키 값을 할당해서 해당 키에 적절한 데이터를 할당하여 return하도록 구현했습니다.

**draw_keypoints**와 **save_samples**: 학습이 끝난 모델로 테스트 이미지를 추론할 때 생긴 결과에서 랜덤으로 이미지를 뽑아 키포인트를 그려서 시각화한 뒤 저장하도록 구현된 함수입니다.

**fix_random_seed**: random seed를 고정하기 위한 함수입니다.

#### train.py

학습이 이뤄지고 마친뒤 추론하는 과정이 담긴 코드입니다.
Detectron2에서 기본으로 horizontal flip transform을 포함하고 있기 때문에 좌우가 뒤집혔을 때 제대로 반영되도록 keypoint_flip_map을 메타데이터에 추가해서 사용했습니다.
Learning Rate는 0.001, iteration은 10000으로 설정하여 학습했습니다.
Coco 키포인트 데이터셋에서는 검증 과정시 스코어 계산을 위해 OKS(Object Keypoint Similarity)를 사용하는데 기존 coco 키포인트에서 사용한 oks sigma 값을 보고 근사해서 넣은 값과 1로 사용했을 때 결과에서 차이가 없어서 1로 사용했습니다.
학습이 끝난 모델로 테스트 이미지를 추론했을 때 간혹 키포인트가 제대로 나오지 않는 이미지가 발생해서 해당 이미지 발생시 0으로 먼저 채워넣고 다른 모델의 추론 값으로 채워넣는 과정을 거쳤습니다.

