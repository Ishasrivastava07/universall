from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Universal Bank Intelligence Hub", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
TARGET = "Personal Loan"
DROP_COLS = ["ID", "ZIP Code"]
BINARY_COLS = ["Securities Account", "CD Account", "Online", "CreditCard"]
EDU_MAP = {1: "Undergrad", 2: "Graduate", 3: "Advanced / Professional"}

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1.2rem; padding-left: 1.4rem; padding-right: 1.4rem; max-width: 100%;}
section[data-testid="stSidebar"] > div {background: #09111f;}
.main-shell {padding: 1.2rem 1.35rem; border-radius: 18px; background: linear-gradient(135deg, #0b1220 0%, #101a2d 100%); border: 1px solid #1e293b; margin-bottom: 1rem;}
.hero-title {font-size: 2.25rem; font-weight: 800; color: #eef4ff; line-height: 1.05; margin: 0 0 0.35rem 0;}
.hero-sub {font-size: 1rem; color: #9fb0c7; margin: 0;}
.tag-row {display:flex; gap:0.45rem; flex-wrap:wrap; margin-top:0.8rem;}
.tag {padding:0.3rem 0.65rem; border-radius:999px; background:#122033; color:#9cd0ff; font-size:0.8rem; border:1px solid #22354e;}
.card {background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%); border: 1px solid #1f2d42; border-radius: 16px; padding: 1rem 1rem 0.9rem 1rem; height: 100%;}
.card-label {color:#93a4bc; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.35rem;}
.card-value {color:#eef4ff; font-size:2rem; font-weight:800; line-height:1;}
.card-note {color:#7f93ad; font-size:0.82rem; margin-top:0.45rem;}
.section-title {font-size:1.2rem; font-weight:700; color:#eef4ff; margin:0.2rem 0 0.9rem 0;}
.insight {background:#0d1626; border:1px solid #1f2d42; border-left:4px solid #60a5fa; padding:0.9rem 1rem; border-radius:12px; color:#d9e6f7; margin:0.4rem 0 1rem 0;}
.offer {background:#0d1626; border:1px solid #204127; border-left:4px solid #22c55e; padding:0.9rem 1rem; border-radius:12px; margin-bottom:0.7rem; color:#e6f6ea;}
.small-muted {color:#8ca0bb; font-size:0.84rem;}
div[data-testid="stMetric"] {background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%); border: 1px solid #1f2d42; padding: 0.9rem 1rem; border-radius: 16px;}
div[data-testid="stMetricLabel"] {color:#93a4bc;}
div[data-testid="stMetricValue"] {color:#eef4ff;}
div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
div[data-baseweb="select"] > div {background:#0d1626; border-color:#21324a;}
button[kind="secondary"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def stratified_split(X, y, test_size=0.25, random_state=42):
    rng = np.random.default_rng(random_state)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0 = max(1, int(len(idx0) * test_size))
    n1 = max(1, int(len(idx1) * test_size))
    test_idx = np.concatenate([idx0[:n0], idx1[:n1]])
    train_idx = np.concatenate([idx0[n0:], idx1[n1:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def confusion(y_true, y_pred):
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def roc_curve_points(y_true, y_score):
    thresholds = np.unique(np.round(y_score, 6))[::-1]
    thresholds = np.concatenate([[1.01], thresholds, [-0.01]])
    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        cm = confusion(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0
        tpr = tp / (tp + fn) if (tp + fn) else 0
        rows.append((fpr, tpr))
    return pd.DataFrame(rows, columns=["FPR", "TPR"]).drop_duplicates()


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1.0 - p * p - (1 - p) * (1 - p)


def mse_impurity(y):
    if len(y) == 0:
        return 0.0
    m = np.mean(y)
    return float(np.mean((y - m) ** 2))


def candidate_thresholds(col):
    vals = np.unique(col)
    if len(vals) <= 12:
        return (vals[:-1] + vals[1:]) / 2
    return np.unique(np.quantile(col, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))


class Node:
    def __init__(self, value=None, prob=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.prob = prob
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeCustom:
    def __init__(self, max_depth=4, min_samples_leaf=25, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.rng_ = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _pick_features(self):
        idx = np.arange(self.n_features_)
        if self.max_features is None or self.max_features >= self.n_features_:
            return idx
        return self.rng_.choice(idx, size=self.max_features, replace=False)

    def _best_split(self, X, y):
        parent = gini_impurity(y)
        best_gain, best_f, best_t = 0.0, None, None
        for f in self._pick_features():
            col = X[:, f]
            for t in candidate_thresholds(col):
                left = col <= t
                right = ~left
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue
                gain = parent - (left.mean() * gini_impurity(y[left]) + right.mean() * gini_impurity(y[right]))
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, float(t)
        return best_gain, best_f, best_t

    def _build(self, X, y, depth):
        prob = float(np.mean(y)) if len(y) else 0.0
        pred = int(prob >= 0.5)
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2 or len(np.unique(y)) == 1:
            return Node(value=pred, prob=prob)
        gain, feat, thr = self._best_split(X, y)
        if feat is None or gain <= 1e-10:
            return Node(value=pred, prob=prob)
        self.feature_importances_[feat] += gain * len(y)
        mask = X[:, feat] <= thr
        return Node(value=pred, prob=prob, feature=feat, threshold=thr,
                    left=self._build(X[mask], y[mask], depth + 1),
                    right=self._build(X[~mask], y[~mask], depth + 1))

    def _predict_prob_one(self, row, node):
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.prob

    def predict_proba(self, X):
        probs = np.array([self._predict_prob_one(r, self.root_) for r in X])
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RegressionTreeCustom:
    def __init__(self, max_depth=2, min_samples_leaf=25, random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root_ = self._build(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _best_split(self, X, y):
        parent = mse_impurity(y)
        best_gain, best_f, best_t = 0.0, None, None
        for f in range(self.n_features_):
            col = X[:, f]
            for t in candidate_thresholds(col):
                left = col <= t
                right = ~left
                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue
                gain = parent - (left.mean() * mse_impurity(y[left]) + right.mean() * mse_impurity(y[right]))
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, float(t)
        return best_gain, best_f, best_t

    def _build(self, X, y, depth):
        val = float(np.mean(y)) if len(y) else 0.0
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2:
            return Node(value=val)
        gain, feat, thr = self._best_split(X, y)
        if feat is None or gain <= 1e-10:
            return Node(value=val)
        self.feature_importances_[feat] += gain * len(y)
        mask = X[:, feat] <= thr
        return Node(value=val, feature=feat, threshold=thr,
                    left=self._build(X[mask], y[mask], depth + 1),
                    right=self._build(X[~mask], y[~mask], depth + 1))

    def _predict_one(self, row, node):
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_one(r, self.root_) for r in X])


class RandomForestCustom:
    def __init__(self, n_estimators=24, max_depth=5, min_samples_leaf=25, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, m = X.shape
        self.trees_ = []
        imps = np.zeros(m)
        max_features = max(1, int(np.sqrt(m)))
        for i in range(self.n_estimators):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            tree = DecisionTreeCustom(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                      max_features=max_features, random_state=self.random_state + i + 1)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            imps += tree.feature_importances_
        self.feature_importances_ = imps / self.n_estimators
        return self

    def predict_proba(self, X):
        probs = np.mean([t.predict_proba(X)[:, 1] for t in self.trees_], axis=0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GradientBoostingCustom:
    def __init__(self, n_estimators=35, learning_rate=0.08, max_depth=2, min_samples_leaf=30, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.base_score_ = math.log(p / (1 - p))
        F = np.full(len(y), self.base_score_)
        self.trees_ = []
        imps = np.zeros(X.shape[1])
        for i in range(self.n_estimators):
            residual = y - sigmoid(F)
            tree = RegressionTreeCustom(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.random_state + i + 1)
            tree.fit(X, residual)
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            imps += tree.feature_importances_
        total = imps.sum()
        self.feature_importances_ = imps / total if total > 0 else imps
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.base_score_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        probs = sigmoid(F)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def chi_square_score(df, feature, target):
    tab = pd.crosstab(df[feature], df[target]).values.astype(float)
    total = tab.sum()
    row_sum = tab.sum(axis=1, keepdims=True)
    col_sum = tab.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    expected[expected == 0] = 1e-9
    return float(((tab - expected) ** 2 / expected).sum())


def find_csv():
    for name in ["UniversalBank.csv", "universalbank.csv"]:
        p = BASE_DIR / name
        if p.exists():
            return p
    files = list(BASE_DIR.glob("*.csv"))
    return files[0] if files else None


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        p = find_csv()
        if p is None:
            raise FileNotFoundError("UniversalBank.csv not found beside app.py")
        df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    df["Experience"] = df["Experience"].clip(lower=0)
    df["Education Label"] = df["Education"].map(EDU_MAP)
    df["Loan Label"] = df[TARGET].map({0: "No", 1: "Yes"})
    df["Income Band"] = pd.cut(df["Income"], bins=[0, 50, 100, 150, 1000], labels=["Low", "Mid", "Upper-Mid", "High"], include_lowest=True)
    df["Age Band"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 70], labels=["21-30", "31-40", "41-50", "51-60", "61-70"], include_lowest=True)
    df["CCAvg Band"] = pd.cut(df["CCAvg"], bins=[-0.01, 1, 3, 6, 100], labels=["Low", "Medium", "High", "Very High"], include_lowest=True)
    df["Wealth Score"] = (0.45 * df["Income"] + 0.25 * df["CCAvg"] * 20 + 0.20 * df["Mortgage"] * 0.2 + 0.10 * df["Education"] * 20).round(1)
    return df


@st.cache_resource
def train_models(df):
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET, "Education Label", "Loan Label", "Income Band", "Age Band", "CCAvg Band", "Wealth Score"]]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test, _, _ = stratified_split(X, y, test_size=0.25, random_state=42)
    models = {
        "Decision Tree": DecisionTreeCustom(max_depth=4, min_samples_leaf=30, random_state=42),
        "Random Forest": RandomForestCustom(n_estimators=24, max_depth=5, min_samples_leaf=25, random_state=42),
        "Gradient Boosting": GradientBoostingCustom(n_estimators=35, learning_rate=0.08, max_depth=2, min_samples_leaf=30, random_state=42),
    }
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy(y_test, pred), 4),
            "Precision": round(precision(y_test, pred), 4),
            "Recall": round(recall(y_test, pred), 4),
            "F1": round(f1_score(y_test, pred), 4),
            "ROC AUC": round(roc_auc(y_test, proba), 4),
        })
        fitted[name] = {"model": model, "cm": confusion(y_test, pred), "roc": roc_curve_points(y_test, proba)}
    metrics = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    best_model = models[best_name]
    full = df.copy()
    probs = best_model.predict_proba(X)[:, 1]
    full["Predicted Probability"] = probs
    full["Predicted Class"] = (probs >= 0.5).astype(int)
    return feature_cols, metrics, fitted, best_name, best_model, full


def persona(row):
    if row["Income"] >= 120 and row["CCAvg"] >= 3:
        return "Affluent Spender"
    if row["Family"] >= 3 and row["Mortgage"] > 0:
        return "Family Borrower"
    if row["Age"] <= 35 and row["Online"] == 1:
        return "Digital Young Professional"
    if row["CD Account"] == 1 or row["Securities Account"] == 1:
        return "Investment-Led Client"
    return "Mass Retail Customer"


def offer(row):
    recs = []
    if row["CD Account"] == 0:
        recs.append("Bundle a CD account with preferential loan pricing")
    if row["Securities Account"] == 0:
        recs.append("Pitch an investment account to deepen relationship value")
    if row["CreditCard"] == 0:
        recs.append("Add a rewards credit card with EMI or cash-back benefits")
    if row["Online"] == 0:
        recs.append("Push digital onboarding for faster loan conversion")
    if not recs:
        recs.append("Offer a pre-approved loan with loyalty pricing")
    return " | ".join(recs[:2])


def acceptance_by(df, col):
    out = df.groupby(col, dropna=False).agg(Customers=(TARGET, "size"), Accepted=(TARGET, "sum")).reset_index()
    out["Acceptance Rate %"] = (out["Accepted"] / out["Customers"] * 100).round(2)
    return out


uploaded = st.sidebar.file_uploader("Upload UniversalBank CSV", type=["csv"])
df = load_data(uploaded)
feature_cols, metrics, fitted, best_name, best_model, scored = train_models(df)

st.sidebar.markdown("### Filters")
income_opts = [x for x in df["Income Band"].dropna().unique()]
edu_opts = sorted(df["Education Label"].dropna().unique())
loan_opts = ["No", "Yes"]
income_sel = st.sidebar.multiselect("Income Band", income_opts, default=income_opts)
edu_sel = st.sidebar.multiselect("Education", edu_opts, default=edu_opts)
loan_sel = st.sidebar.multiselect("Personal Loan", loan_opts, default=loan_opts)

view = df[df["Income Band"].isin(income_sel) & df["Education Label"].isin(edu_sel) & df["Loan Label"].isin(loan_sel)].copy()

loan_rate = view[TARGET].mean() * 100
high_income_share = (view["Income"] >= 100).mean() * 100
cd_share = view["CD Account"].mean() * 100
online_share = view["Online"].mean() * 100
wealth_avg = view["Wealth Score"].mean()

best_metric = metrics.iloc[0].to_dict()

st.markdown("""
<div class="main-shell">
  <div class="hero-title">Universal Bank Intelligence Hub</div>
  <p class="hero-sub">A cleaner client-ready view of loan conversion potential, segment behavior, model output, and cross-sell actions.</p>
  <div class="tag-row">
    <div class="tag">Personal Loan Prediction</div>
    <div class="tag">Cross-Sell Targeting</div>
    <div class="tag">Client-Friendly Layout</div>
    <div class="tag">No heavy dependencies</div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
for col, label, value, note in [
    (c1, "Customers", f"{len(view):,}", "Filtered audience size"),
    (c2, "Loan Acceptance", f"{loan_rate:.1f}%", "Observed conversion rate"),
    (c3, "High Income Share", f"{high_income_share:.1f}%", "Income ≥ 100k"),
    (c4, "Digital Usage", f"{online_share:.1f}%", "Online banking adoption"),
    (c5, "Avg Wealth Score", f"{wealth_avg:.1f}", "Composite affluence indicator"),
]:
    col.markdown(f"<div class='card'><div class='card-label'>{label}</div><div class='card-value'>{value}</div><div class='card-note'>{note}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["Descriptive", "Diagnostic", "Predictive", "Prescriptive"])

with t1:
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        st.markdown("<div class='section-title'>Descriptive Snapshot</div>", unsafe_allow_html=True)
        split = view["Loan Label"].value_counts().rename_axis("Outcome").to_frame("Customers")
        st.bar_chart(split)
        st.dataframe(split.reset_index(), use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Descriptive Highlights</div>", unsafe_allow_html=True)
        top_edu = acceptance_by(view, "Education Label").sort_values("Acceptance Rate %", ascending=False).iloc[0]
        top_band = acceptance_by(view, "Income Band").sort_values("Acceptance Rate %", ascending=False).iloc[0]
        st.markdown(f"<div class='insight'><b>Best education segment:</b> {top_edu['Education Label']} with {top_edu['Acceptance Rate %']:.1f}% acceptance.<br><br><b>Best income segment:</b> {top_band['Income Band']} with {top_band['Acceptance Rate %']:.1f}% acceptance.<br><br><b>Best model:</b> {best_name} with ROC AUC {best_metric['ROC AUC']:.3f}.</div>", unsafe_allow_html=True)
        own = []
        for col in BINARY_COLS:
            grp = acceptance_by(view, col)
            yes_rate = float(grp.loc[grp[col] == 1, "Acceptance Rate %"].iloc[0]) if 1 in grp[col].values else 0
            no_rate = float(grp.loc[grp[col] == 0, "Acceptance Rate %"].iloc[0]) if 0 in grp[col].values else 0
            own.append((col, round(yes_rate - no_rate, 2)))
        own_df = pd.DataFrame(own, columns=["Product", "Acceptance Lift %"]).sort_values("Acceptance Lift %", ascending=False)
        st.dataframe(own_df, use_container_width=True, hide_index=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Acceptance by Education</div>", unsafe_allow_html=True)
        edu = acceptance_by(view, "Education Label").set_index("Education Label")[["Acceptance Rate %"]]
        st.bar_chart(edu)
        st.dataframe(edu.reset_index(), use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Acceptance by Income Band</div>", unsafe_allow_html=True)
        inc = acceptance_by(view, "Income Band").set_index("Income Band")[["Acceptance Rate %"]]
        st.bar_chart(inc)
        st.dataframe(inc.reset_index(), use_container_width=True, hide_index=True)

with t2:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Diagnostic Drivers</div>", unsafe_allow_html=True)
        age = acceptance_by(view, "Age Band").set_index("Age Band")[["Acceptance Rate %"]]
        st.line_chart(age)
        st.dataframe(age.reset_index(), use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Diagnostic Spend View</div>", unsafe_allow_html=True)
        ccb = acceptance_by(view, "CCAvg Band").set_index("CCAvg Band")[["Acceptance Rate %"]]
        st.bar_chart(ccb)
        st.dataframe(ccb.reset_index(), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Diagnostic Cross-Sell Matrix</div>", unsafe_allow_html=True)
    matrix = []
    for col in BINARY_COLS:
        grp = acceptance_by(view, col)
        yes_rate = float(grp.loc[grp[col] == 1, "Acceptance Rate %"].iloc[0]) if 1 in grp[col].values else 0
        no_rate = float(grp.loc[grp[col] == 0, "Acceptance Rate %"].iloc[0]) if 0 in grp[col].values else 0
        matrix.append({"Product": col, "Has Product Rate %": round(yes_rate, 2), "No Product Rate %": round(no_rate, 2), "Lift %": round(yes_rate - no_rate, 2)})
    matrix_df = pd.DataFrame(matrix).sort_values("Lift %", ascending=False)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Diagnostic Drivers</div>", unsafe_allow_html=True)
    corr_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard", TARGET]
    corr = view[corr_cols].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    chi_df = pd.DataFrame({"Feature": BINARY_COLS + ["Education"], "Chi-Square": [round(chi_square_score(view, c, TARGET), 2) for c in BINARY_COLS + ["Education"]]}).sort_values("Chi-Square", ascending=False)
    x, y = st.columns(2, gap="large")
    x.dataframe(corr_df, use_container_width=True, hide_index=True)
    y.dataframe(chi_df, use_container_width=True, hide_index=True)

with t3:
    a, b = st.columns([1.05, 1], gap="large")
    with a:
        st.markdown("<div class='section-title'>Predictive Model Scorecard</div>", unsafe_allow_html=True)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.markdown(f"<div class='insight'><b>{best_name}</b> leads on ROC AUC and is the recommended model for the friend-facing demo version.</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section-title'>Predictive Confusion Matrix</div>", unsafe_allow_html=True)
        cm = pd.DataFrame(fitted[best_name]["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm, use_container_width=True)
        st.markdown("<div class='small-muted'>Rows are actual outcomes; columns are predicted outcomes.</div>", unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Predictive Feature Importance</div>", unsafe_allow_html=True)
        imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).round(4)
        st.bar_chart(imp.head(10))
        imp_df = imp.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        st.dataframe(imp_df.head(10), use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Predictive ROC View</div>", unsafe_allow_html=True)
        st.line_chart(fitted[best_name]["roc"].set_index("FPR"))
        st.markdown("<div class='small-muted'>Closer to the upper-left corner means stronger discrimination.</div>", unsafe_allow_html=True)

with t4:
    leads = scored[scored["Predicted Class"] == 1].copy()
    leads["Persona"] = leads.apply(persona, axis=1)
    leads["Recommended Offer"] = leads.apply(offer, axis=1)
    leads = leads.sort_values("Predicted Probability", ascending=False)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='section-title'>Prescriptive Persona Mix</div>", unsafe_allow_html=True)
        persona_df = leads["Persona"].value_counts().to_frame("Customers")
        st.bar_chart(persona_df)
        st.dataframe(persona_df.reset_index().rename(columns={"index": "Persona"}), use_container_width=True, hide_index=True)
    with b:
        st.markdown("<div class='section-title'>Prescriptive Lead List</div>", unsafe_allow_html=True)
        top_leads = leads[["Age", "Income", "Family", "CCAvg", "Education Label", "Predicted Probability"]].head(15).copy()
        top_leads["Predicted Probability"] = top_leads["Predicted Probability"].round(4)
        st.dataframe(top_leads, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Prescriptive Recommendations</div>", unsafe_allow_html=True)
    for _, row in leads.head(8).iterrows():
        st.markdown(f"<div class='offer'><b>{persona(row)}</b><br>Probability: {row['Predicted Probability']:.1%}<br>{offer(row)}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Prescriptive Simulator</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3, gap="large")
    with s1:
        age = st.slider("Age", 21, 70, 35)
        experience = st.slider("Experience", 0, 45, 10)
        income = st.slider("Income ($000)", 5, 250, 100)
        family = st.slider("Family", 1, 4, 2)
    with s2:
        ccavg = st.slider("CCAvg ($000)", 0.0, 12.0, 2.0, 0.1)
        education = st.selectbox("Education", [1, 2, 3], format_func=lambda x: EDU_MAP[x])
        mortgage = st.slider("Mortgage ($000)", 0, 650, 50)
    with s3:
        securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
        cd = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
        online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: "Yes" if x else "No")
        credit = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")

    sample = pd.DataFrame([{
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities Account": securities,
        "CD Account": cd,
        "Online": online,
        "CreditCard": credit,
    }])
    prob = float(best_model.predict_proba(sample[feature_cols].to_numpy(dtype=float))[:, 1][0])
    sample_row = sample.iloc[0].copy()
    st.markdown(f"<div class='offer'><b>Predicted acceptance probability:</b> {prob:.1%}<br><b>Persona:</b> {persona(sample_row)}<br><b>Recommended action:</b> {offer(sample_row)}</div>", unsafe_allow_html=True)

st.caption("Friend-ready edition: cleaner layout, tighter spacing, simplified story flow, and dependency-light deployment.")
