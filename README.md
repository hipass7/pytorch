# VGG19를 이용한 CIFAR-10 데이터 분류
1. 10개 class 모두 5000개의 데이터를 사용하여 학습
2. 5개의 major class (5000개) : 5개의 minor class (500개)를 사용하여 학습
3. 1개의 major class (5000개) : 9개의 minor class (500개)를 사용하여 학습
4. 3번의 상황에서 SMOTE를 사용한 augmentation 이후 학습
