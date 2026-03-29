from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from feature_engineering import FeatureEngineering


num_features = [
    "Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck",
    "Cabin_num","TotalSpending"
]

cat_features = [
    "HomePlanet","CryoSleep","Destination","VIP",
    "Deck","Side","Age_group"
]


def build_pipeline():

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    model = Pipeline([
        ("feature_engineering", FeatureEngineering()),  # 🔥 INI WAJIB
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model


def train_model(X, y):
    model = build_pipeline()
    model.fit(X, y)
    return model