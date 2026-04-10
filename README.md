#  IMBK_Bank_Customer_Churn_ML

🏦 IMBK_Bank_Customer_Churn_ML
네, 맞습니다! 점(Bullet point)이나 구조가 깨지면 깃허브에서 일일이 수정하기 번거로우시죠. 이번에는 **마크다운(Markdown) 문법을 완벽하게 적용**해서, 복사해서 붙여넣기만 하면 바로 전문적인 포트폴리오처럼 보일 수 있도록 아주 깔끔하고 "이쁘게" 정리해 드릴게요.

---

# 📑 README.md

# 🏦 IMBK_Bank_Customer_Churn_ML
> **은행 고객 이탈 데이터를 활용한 분류 머신러닝 모델 개발 및 데이터 기반 비즈니스 인사이트 도출 프로젝트**

---

## 1. 🚀 프로젝트명
**고객 이탈 분류 ML 및 인사이트 분석**
* 본 프로젝트는 고객의 금융 행동 데이터를 분석하여 이탈 여부를 예측하고, SHAP을 활용한 사후 분석으로 구체적인 리텐션 전략을 제안하는 것을 목적으로 합니다.

## 2. 📅 기간
* **2026년 4월 10일** (머신러닝 컴페티션)

## 3. 🛠 기술 스택 (Tech Stack)
* **언어 및 환경:** `Python 3.x`, `Jupyter Notebook`
* **데이터 분석:** `Pandas`, `NumPy`
* **시각화:** `Matplotlib`, `Seaborn`
* **머신러닝:** `PyCaret(AutoML)`, `Scikit-learn`, `LightGBM`, `CatBoost`, `AdaBoost`
* **최적화 및 해석:** `Optuna` (Hyperparameter Tuning), `SHAP` (XAI)

## 4. 📊 데이터 출처
* **Source:** [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
* **Dataset Info:** 10,000 rows × 12 columns
* **Target:** `churn` (0: 유지, 1: 이탈)

## 5. ⚙️ 데이터 전처리
* **데이터 정제:** 분석과 무관한 고유 식별자(`customer_id`) 제거
* **결측치 처리:** 전수 검사 결과 결측치 없음 확인
* **범주형 변수 처리:** `country`, `gender` 피처에 대한 `Label Encoding` 적용
* **데이터 스케일링:** 모델 성능 최적화를 위해 `StandardScaler` 적용
* **데이터 분할:** `Stratified Split`을 통해 타겟 비율을 유지하며 학습/검증 데이터 8:2 분할

## 6. 🔍 EDA 및 해석
* **이탈 현황:** 유지 고객 대비 이탈 고객의 비율이 낮아 클래스 불균형 존재 확인.
* **연령별 패턴:** 50대 고객층에서 이탈률이 정점을 찍는 양상을 보임 (특화 상품 필요성 대두).
* **국가별 비교:** France/Germany 대비 Spain 고객의 이탈자 수가 현저히 적음 (국가별 정책 차이 분석 필요).
* **상관성 분석:** 피처 간 상관계수 히트맵을 통해 `age`, `balance` 등의 주요 변수 영향력 사전 파악.

## 7. 🤖 모델링 전략
### 💎 AutoML & Tuning
* **Model Selection:** `PyCaret`을 활용하여 F1-Score 기준 상위 4개 모델(AdaBoost, LGBM, GBC, CatBoost) 선정.
* **Hyperparameter Tuning:** `Optuna`를 이용한 베이지안 최적화로 각 모델별 최적의 파라미터 도출.

### 🔗 Stacking Ensemble
* **Base Models:** `AdaBoost`, `LGBM`, `GBC`, `CatBoost`
* **Meta Model:** `LogisticRegression`
* **Performance:** **최종 F1-Score: 0.6129**, **Accuracy: 0.8705**

### 💡 Model Interpretation (SHAP Value)
* `products_number`와 `age`가 이탈 결정에 가장 결정적인 영향을 미치는 핵심 변수임을 도출.
* 독일(Germany) 거주 여부와 활성 회원(Active Member) 여부가 주요 판단 근거로 작용.

## 8. 🎯 인사이트 제안
1. **고위험군 타겟 관리:** 상품 이용 수가 비정상적으로 높거나 낮은 **50대 고객**을 대상으로 전담 상담원 배정 및 자산 관리 컨설팅 제공.
2. **활동성 강화 캠페인:** 비활성 고객(Inactive Member)의 이탈을 막기 위해 휴면 고객 대상 수수료 면제 및 개인화된 알림 이벤트 실행.
3. **VIP 잔액 관리 프로그램:** 잔액이 높은 우량 고객의 이탈이 관찰됨에 따라, 예금 잔액에 비례한 **차등 등급제(VIP)** 및 전용 혜택 제공으로 락인(Lock-in) 효과 강화.

## 9. 📚 Reference
* [Kaggle Dataset: Bank Customer Churn](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
* [PyCaret 공식 문서](https://pycaret.org/)
* [SHAP Library Documentation](https://shap.readthedocs.io/)

---
### 👤 Author
* **신유진 (Shin Yu-jin)** - *머신러닝 모델 개발 및 데이터 분석*

---

### 💡 팁
1. 깃허브 리포지토리 메인 화면의 **`Add a README`** 버튼을 누릅니다.
2. 위 박스 안의 내용을 그대로 **복사(Ctrl+C)해서 붙여넣기(Ctrl+V)** 하세요.
3. `Preview` 탭을 눌러 이쁘게 나오는지 확인하고 `Commit` 하시면 끝납니다!
