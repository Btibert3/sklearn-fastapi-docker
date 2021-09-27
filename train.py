"""Train the sklearn model"""

# imports
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import datetime
import brock


# get the data -- sklearn bunch object, has data and target names
CATEGORIES = brock.CATEGORIES
train = fetch_20newsgroups(subset='train',shuffle=True, random_state=42, categories=CATEGORIES)
test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42, categories=CATEGORIES)

# breakout into train/test groups
X_train = train.data
y_train = train.target
X_test = test.data
y_test = test.target

# build the pipeline
steps = [('vecs', TfidfVectorizer(min_df = 20, max_features=12000, ngram_range=(1,2))),
         ('pca', TruncatedSVD(100)),
         ('scale', MinMaxScaler()),
         ('clf', MultinomialNB())]
pipeline = Pipeline(steps)

# fit the model
pipeline.fit(X_train, y_train)

# test the model and report
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

# save the model if better than a logic that could obviously be based on standards
# from previous models
acc = accuracy_score(y_test, preds)
THRESHOLD = .1  # this should be dynamic from past runs
if acc > THRESHOLD:
    print("model improved the previous models/baseline")
    # TS = int(datetime.datetime.now().timestamp())   # to help with timestamping for archiving
    # FNAME = f"model/model-{TS}.joblib"
    TS = "final"
    joblib.dump(pipeline, f"model/model-{TS}.joblib" )

