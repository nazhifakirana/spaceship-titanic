from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df[['Deck','Cabin_num','Side']] = df['Cabin'].str.split('/', expand=True)
        df['Cabin_num'] = pd.to_numeric(df['Cabin_num'], errors='coerce')

        spending_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
        df['TotalSpending'] = df[spending_cols].sum(axis=1)

        df['Age_group'] = pd.cut(
            df['Age'],
            bins=[0,12,18,30,50,100],
            labels=['Child','Teen','Young_Adult','Adult','Senior']
        ).astype(str)

        return df