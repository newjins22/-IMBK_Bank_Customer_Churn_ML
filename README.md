#  IMBK_Bank_Customer_Churn_ML
> 은행 고객 이탈 데이터를 활용한 분류 머신러닝 모델 개발 및 데이터 기반 비즈니스 인사이트 도출 프로젝트

---

## 1. 프로젝트명
**고객 이탈 분류 ML 및 인사이트 분석**
* 본 프로젝트는 고객의 금융 행동 데이터를 분석하여 이탈 여부를 예측하고, SHAP을 활용한 사후 분석으로 구체적인 리텐션 전략을 제안하는 것을 목적으로 함.

## 2. 기간
* **2026년 4월 10일** (머신러닝 컴페티션)

## 3. 기술 스택 (Tech Stack)
* **언어 및 환경:** `Python 3.x`, `Jupyter Notebook`
* **데이터 분석:** `Pandas`, `NumPy`
* **시각화:** `Matplotlib`, `Seaborn`
* **머신러닝:** `PyCaret(AutoML)`, `Scikit-learn`, `LightGBM`, `CatBoost`, `AdaBoost`
* **최적화 및 해석:** `Optuna` (Hyperparameter Tuning), `SHAP` (XAI)

## 4. 데이터 출처
* **Source:** [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
* **Dataset Info:** 10,000 rows × 12 columns
* **Target:** `churn` (0: 유지, 1: 이탈)

## 5. 데이터 전처리
* **데이터 정제:** 분석과 무관한 고유 식별자(`customer_id`) 제거
* **결측치 처리:** 전수 검사 결과 결측치 없음 확인
* **범주형 변수 처리:** `country`, `gender` 피처에 대한 `Label Encoding` 적용
* **데이터 스케일링:** 모델 성능 최적화를 위해 `StandardScaler` 적용
* **데이터 분할:** `Stratified Split`을 통해 타겟 비율을 유지하며 학습/검증 데이터 8:2 분할


## 6. EDA 및 해석 (Exploratory Data Analysis)
데이터의 핵심 특성을 파악하기 위해 다각도로 시각화를 진행하였으며, 주요 분석 결과는 다음과 같음.

### **① 이탈 여부 분포 (Churn Distribution)**
* **현황:** 유지 고객(Stay)의 비율이 이탈 고객(Exit)보다 압도적으로 높음.
* **해석:** 전형적인 클래스 불균형(Imbalanced Data) 특성을 보이며, 모델 평가 시 단순 정확도(`Accuracy`)보다는 정밀도와 재현율을 동시에 고려한 `F1-Score`를 주요 지표로 삼아야 함을 시사함.
<img width="832" height="641" alt="image" src="https://github.com/user-attachments/assets/e7552688-2649-46dd-a512-d78a8d4cd73c" />


### **② 연령대별 평균 이탈률 (Churn Rate by Age Group)**
* **현황:** 30대부터 이탈률이 급격히 상승하여 50대에서 정점을 찍고, 70대 이후로 감소하는 종형 곡선(Bell Curve) 형태를 보임.
* **해석:** 50대 고객층은 자산 유동성이 크고 상품 전환이 활발한 '고위험군'으로 판단됨. 이들을 타겟으로 한 전용 금융 상품이나 맞춤형 유지 혜택 제안이 비즈니스적으로 시급함.
<img width="840" height="643" alt="image" src="https://github.com/user-attachments/assets/5287d68b-7cad-44a0-b669-9a4b915cdff5" />

### **③ 국가별 이탈자 합계 (Churn Count by Country)**
* **현황:** 프랑스(0)와 독일(1)에 비해 스페인(2)의 이탈자 수가 현저히 적음.
* **해석:** 스페인 지점의 독자적인 고객 관리 전략이나 현지 시장의 특수성(낮은 경쟁률 등)을 심층 분석하여, 이탈률이 높은 타 국가 지점에 벤치마킹할 가치가 있음.
<img width="831" height="647" alt="image" src="https://github.com/user-attachments/assets/a00b31f4-591b-4d28-aa8d-d7185d68820a" />

### **④ 상관관계 히트맵 (Feature Correlation Heatmap)**
* **현황:** `Age`와 `Churn` 사이에 유의미한 양의 상관관계가 관찰되며, 그 외 변수들은 서로 독립적인 경향을 보임.
* **해석:** 나이가 이탈 예측의 핵심 인자임을 재확인하였으며, 변수 간 다중공선성(Multicollinearity) 문제가 적어 모든 피처를 모델링에 안정적으로 활용할 수 있음을 확인함.
<img width="665" height="549" alt="image" src="https://github.com/user-attachments/assets/f9190f4e-d829-4d84-b055-9e6683e7f2f3" />



## 7. 모델링 전략
### AutoML & Tuning
* **Model Selection:** `PyCaret`을 활용하여 F1-Score 기준 상위 4개 모델(AdaBoost, LGBM, GBC, CatBoost) 선정.
* **Hyperparameter Tuning:** `Optuna`를 이용한 베이지안 최적화로 각 모델별 최적의 파라미터 도출.
<img width="693" height="402" alt="image" src="https://github.com/user-attachments/assets/1464bbf7-cdfa-42e4-b10a-fd53ec4a231c" />


### Stacking Ensemble
* **Base Models:** `AdaBoost`, `LGBM`, `GBC`, `CatBoost`
* **Meta Model:** `LogisticRegression`
* **Performance:** **최종 F1-Score: 0.6129**, **Accuracy: 0.8705**

### Model Interpretation (SHAP Value)
학습된 모델이 어떤 근거로 고객의 이탈(Churn)을 예측했는지 `SHAP(SHapley Additive exPlanations)`를 통해 분석한 결과

#### **[핵심 변수 분석]**
* **`products_number` (상품 수):** 모델에 가장 큰 영향을 미치는 변수. 특정 임계값을 넘어가면 이탈 가능성이 급격히 커지며, 일반적인 경우 상품 수가 많으면 유지, 적으면 이탈하는 경향을 보임.
* **`age` (나이):** 나이가 많을수록 이탈 가능성이 높고, 어릴수록 낮음. 다만 고연령층 중에서도 일부 유지 성향을 보이는 예외 케이스가 존재함.
* **`active_member` (활성 회원):** 활성 고객은 유지, 비활성 고객은 이탈로 판단하는 뚜렷한 경향을 보임.
* **`balance` (계좌 잔액):** 잔액이 많을수록 오히려 이탈에 가까워지는 역설적인 패턴이 관찰됨.

#### **[기타 변수 및 영향력]**
* **`country` & `gender`:** 프랑스 고객은 유지, 독일 고객은 이탈에 가까움. 성별의 경우 여성이 이탈, 남성이 유지에 가깝게 나타나나 전체적인 영향력 폭은 좁은 편임.
* **`credit_score` & `tenure`:** 신용도가 낮거나 이용 기간이 짧을수록 이탈에 가깝지만, 분포가 0 근처에 밀집되어 있어 결정적인 요인은 아닌 것으로 분석됨.
* **`estimated_salary` & `credit_card`:** 점들이 뚜렷한 양상을 보이지 않거나 분포가 매우 좁아, 모델이 이탈 여부를 결정할 때 미치는 영향이 미미함.

> **결론적으로,** 모델은 상품 수(`products_number`)와 나이(`age`)의 분포를 가장 중요하게 참조하여 이탈 여부 결정
 <img width="940" height="675" alt="SHAP value" src="https://github.com/user-attachments/assets/cc5d4f7a-3b8e-4c92-858b-ec40e0dbf61b" />

## 8. 인사이트 제안
분석 결과를 바탕으로 고객 이탈 방지를 위한 3가지 핵심 비즈니스 전략 제안

### **① 고위험군 집중 케어 (Targeting Age & Products)**
* **현황:** 특정 개수 이상의 상품을 보유한 50대 이상 고객의 이탈 위험이 가장 높음.
* **전략:** 해당 조건에 부합하는 고객을 '집중 관리군'으로 분류하고, 전담 상담원을 배정하여 맞춤형 자산 관리 컨설팅(CRM)을 제공함으로써 락인(Lock-in) 효과 강화.

### **② 고객 활동성 강화 이벤트 (Active Member Boosting)**
* **현황:** 비활성 상태가 이탈의 주요 판단 근거 중 하나로 작용함.
* **전략:** 일정 기간 거래가 없는 휴면 고객을 대상으로 '수수료 면제' 또는 '맞춤형 혜택 알림' 프로모션을 실행하여 앱/서비스 접속 및 활동성을 부여함.

### **③ 잔액 기반 로열티 프로그램 (Balance-Based Retention)**
* **현황:** 잔액이 높은 우량 고객일수록 오히려 이탈 경향이 높게 나타남.
* **전략:** 고액 잔액 유지 고객을 위해 자산 규모에 비례한 차등 등급제(VIP 프로그램)를 신설하고, 금리 우대나 세무 상담 등 타사로 이동하기 어려운 독보적인 혜택을 제공함.


## 9. Reference
* [Kaggle Dataset: Bank Customer Churn](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
* [PyCaret 공식 문서](https://pycaret.org/)
* [SHAP Library Documentation](https://shap.readthedocs.io/)

---
> 본 프로젝트는 머신러닝의 풀 프로세스 이해와 데이터 해석을 통한 설명력 강화를 목적으로 수행되었습니다.
