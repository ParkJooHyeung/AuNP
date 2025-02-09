#전기공학머신러닝실습  
###예비보고서_Ex1
  
전기공학과  
2019732023  
박주형    
  
---
#개요  
1. 이론  
+ 머신러닝
2. 실험기기   
+ Google Colab
3. 예비보고서 문제   
4. 실험순서

---   
#이론
###머신러닝의 정의  
  
####머신러닝은 학습과 개선을 토대로 컴퓨터에 원시적인 프로그래밍 대신, 컴퓨터가 스스로 방대한 데이터를 바탕으로 학습하고 경험을 통해 어떠한 문제에 대해 해결하는데 있어서 훈련하는 것에 의의를 둔 인공지능(AI)의 하위 집합 기술이다. 이것은 방식에 따라 세가지의 학습방법이 존재한다.  

* _지도 학습(Supervised Learning)_   
  데이터에서 반복적으로 학습하는 알고리즘을 사용하여 컴퓨터가 어디를 찾아봐야 하는 지를 명시적으로 프로그래밍하지 않고도 숨겨진 통찰력을 찾을 수 있도록 하는 데이터 분석 방법이다.  
  ![지도 학습 자료](https://www.tibco.com/sites/tibco/files/media_entity/2020-09/supervised-learning-diagram.svg)
* _비지도 학습(Unsupervised Learning)_
  비지도 학습은 학습 알고리즘에 결과물이라고 할 수 있는 출력을 미리 제공하지 않고 인공지능(AI)이 입력 세트에서 패턴과 상관관계를 찾아내야 하는 머신러닝 알고리즘이다.  
  ![비지도 학습 자료](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSensFuFxdjyIo7Qhd4ssEPZPBfnEITanfwrpXXsIilw&s)
* _강화 학습(Reinforcement Learning)_  
  데이터 점 또는 경험은 훈련하는 동안 환경과 소프트웨어 에이전트 간의 시행착오 상호작용을 통해 수집되고 강화학습의 이런 점은 지도 및 비지도 머신러닝에서는 필요한 훈련 전 데이터 수집, 전처리 및 레이블 지정에 대한 필요성을 해소하기 때문에 중요하다. 이는 실질적으로 적절한 인센티브가 주어지면 강화학습 모델은 인간의 개입 없이 학습 행동을 자체적으로 시작할 수 있다는 것을 의미한다.   
![강화 학습 자료](https://kr.mathworks.com/discovery/reinforcement-learning/_jcr_content/mainParsys3/discoverysubsection/mainParsys/image.adapt.full.medium.png/1675723392074.png)
---   
#실험기기   
##Google Colab   
####웹을 통해 텍스트 기반의 코딩을 할 수 있도록 제공되는 edito로써 Python을 바탕으로 빠른 연산을 가능케 하는 클라우드 컴퓨팅이다.   
