#  IMBK_Bank_Customer_Churn_ML

🏦 IMBK_Bank_Customer_Churn_ML
1. 프로젝트명
- 은행 고객 이탈 분류 머신러닝 모델 개발 및 데이터 기반 인사이트 분석

2. 기간
- 2026년 4월 10일 (머신러닝 컴페티션)

3. 기술 스택
Language: Python

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn (Stacking, RandomForest, LogisticRegression, SVM, KNN)

AutoML & Tuning: PyCaret, Optuna

XAI (Explainable AI): SHAP (Shapley Additive Explanations)

4. 데이터 정보
- 출처: Kaggle Bank Customer Churn Dataset
- 링크: https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data
- 규모: 10,000 Rows, 12 Columns

주요 변수: credit_score, country, gender, age, balance, products_number, active_member, churn (Target) 등

5. 데이터 전처리
- 결측치 확인: 결측치 없음 확인.

- 피처 엔지니어링: 모델 학습에 불필요한 고유 식별 번호(customer_id) 제거.

- 인코딩: 범주형 변수(country, gender)에 대해 Label Encoding 적용.

- 데이터 분할: churn 타겟 비율을 유지하기 위해 stratify 옵션을 사용하여 Train/Valid 데이터 8:2 분할.

6. EDA 및 해석
연령대별 이탈률: 50대 이상 고연령층 고객군에서 이탈 위험이 상대적으로 높게 나타남.

상품 이용수: 특정 개수 이상의 상품을 이용 중인 고객들의 패턴 분석 필요.

활동성: 거래가 없는 휴면 고객(Inactive Member)의 이탈 경향성 확인.

잔액 상태: 잔액이 높은 고객층에서도 이탈이 발생하는 역설적 패턴 관찰.

7. 모델링 전략 (AutoML to Stacking)
AutoML (PyCaret): F1-Score를 기준으로 성능이 우수한 상위 모델 후보군 선정.

Hyperparameter Tuning: Optuna를 활용하여 개별 모델(RF, SVM, KNN)의 최적 파라미터 탐색.

Stacking Pipe:

Base Models: RandomForest, SVM, KNN

Meta Model: Logistic Regression

성능 지표: 최종 모델은 F1-Score 0.6129, Accuracy 0.8705 달성.

사후 분석 (SHAP): TreeExplainer를 사용하여 모델의 의사결정에 기여도가 높은 핵심 피처 파악.

8. 인사이트 및 비즈니스 제안
고위험군 집중 관리: 특정 개수 이상의 상품을 이용하는 50대 이상 고객을 고위험군으로 분류하고 전담 상담원을 통한 자산 관리 서비스 제공.

활동성 강화 이벤트: 일정 기간 거래가 없는 휴면 고객에게 수수료 면제 또는 알림 서비스를 제공하여 리텐션 유도.

등급제 기반 차등 서비스: 높은 잔액 보유 고객의 이탈을 막기 위해 예금 잔액에 비례한 VIP 등급제를 시행하고 차별화된 혜택 제공.

9. Reference
Kaggle: Bank Customer Churn Dataset

Scikit-learn Documentation

SHAP Documentation

PyCaret Classification Module

본 프로젝트는 머신러닝의 풀 프로세스 이해와 데이터 해석을 통한 설명력 강화를 목적으로 수행되었습니다.
