"""Bloom's-level + difficulty prediction trained on Dataset.xlsx."""
import re, joblib, pathlib
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA = pathlib.Path(__file__).parent / "data"
MODEL = DATA / "models.joblib"
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
DIFFS = ["Easy", "Medium", "Hard"]


def load_df():
    df = pd.read_excel(DATA / "Dataset.xlsx")
    df.columns = ["question", "verb", "level", "difficulty"]
    return df.drop_duplicates("question").reset_index(drop=True)


def first_verb(q):
    m = re.match(r"\s*([A-Za-z]+)", q)
    return m.group(1).lower() if m else ""


def verb_features(df):
    """Level distribution of every leading verb seen in the training data."""
    t = df.assign(v=df.question.str.split().str[0].str.lower())
    return (pd.crosstab(t.v, t.level, normalize="index")
            .reindex(columns=LEVELS, fill_value=0))


class Predictor:
    def __init__(self):
        self.w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        self.lvl = LogisticRegression(max_iter=2000, C=8)
        self.dif = LogisticRegression(max_iter=2000, C=4)
        self.vf = None

    def _X(self, qs, fit=False):
        f = (lambda v, x: v.fit_transform(x)) if fit else (lambda v, x: v.transform(x))
        Xw, Xc = f(self.w, qs), f(self.c, qs)
        # the leading verb is the strongest signal in Bloom wording
        V = np.array([self.vf.loc[first_word(q)].values if first_word(q) in self.vf.index
                      else np.zeros(6) for q in qs]) * 2
        L = np.array([[len(q.split()) / 40, q.count(",") / 4, (" and " in q.lower()) * 1.0] for q in qs])
        return hstack([Xw, Xc, csr_matrix(V), csr_matrix(L)]).tocsr()

    def fit(self, df):
        self.vf = verb_features(df)
        X = self._X(df.question.tolist(), fit=True)
        self.lvl.fit(X, df.level)
        # difficulty model also sees the (true) level one-hot
        self.dif.fit(hstack([X, csr_matrix(pd.get_dummies(df.level).reindex(columns=LEVELS, fill_value=0).values * 1.0)]).tocsr(), df.difficulty)
        return self

    def predict(self, q):
        X = self._X([q])
        lp = self.lvl.predict_proba(X)[0]
        lv = dict(zip(self.lvl.classes_, lp))
        level = max(lv, key=lv.get)
        oh = csr_matrix([[1.0 if l == level else 0.0 for l in LEVELS]])
        dp = self.dif.predict_proba(hstack([X, oh]).tocsr())[0]
        dv = dict(zip(self.dif.classes_, dp))
        diff = max(dv, key=dv.get)
        return {"level": level, "level_confidence": round(float(lv[level]), 3),
                "level_probs": {k: round(float(lv.get(k, 0)), 3) for k in LEVELS},
                "difficulty": diff, "difficulty_confidence": round(float(dv[diff]), 3),
                "difficulty_probs": {k: round(float(dv.get(k, 0)), 3) for k in DIFFS},
                "action_word": first_verb(q)}


def first_word(q):
    return q.split()[0].lower() if q.split() else ""


def train(report=True):
    df = load_df()
    p = Predictor().fit(df)
    joblib.dump(p, MODEL, compress=3)
    if report:
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        # honest CV: re-fit the vectorizers/verb table inside each fold
        accs_l, accs_d = [], []
        for tr, te in cv.split(df, df.level):
            m = Predictor().fit(df.iloc[tr])
            pr = [m.predict(q) for q in df.question.iloc[te]]
            accs_l.append(np.mean([a["level"] == b for a, b in zip(pr, df.level.iloc[te])]))
            accs_d.append(np.mean([a["difficulty"] == b for a, b in zip(pr, df.difficulty.iloc[te])]))
        print(f"{len(df)} questions | Bloom level CV acc {np.mean(accs_l):.3f} | difficulty CV acc {np.mean(accs_d):.3f}")
    return p


def get_predictor():
    if MODEL.exists():
        return joblib.load(MODEL)
    return train(report=False)


if __name__ == "__main__":
    import ml  # so the pickle references ml.Predictor, not __main__.Predictor
    ml.train()
