import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import(
    r2_score, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix
)

class AnalystAgent:
    """
    Core engine reponsible for performing data analysis,
    visualization, encoding and model building.
    """

    def __init__(self,df):
        """
        Initialize the AnalystAgent with dataset.

        Args:
            df (pd.DataFrame): Input dataset
        """
        self.df = df.copy()
        self.results = {}
        self.is_encoded = False

    def preview_data(self):
        """
        Returns first 5 rows of dataset
        """
        return self.df.head()

    def null_analysis(self):
        """
        Returns count of missing values per column.
        """
        return self.df.isnull().sum()
    
    def describe_data(self):
        """
        Returns statistical summary of numerical columns.
        """
        return self.df.describe()
    
    def histogram(self,column):
        """
        Generates interactive histogram for a numeric column.
        Args:
            column (str): Column name
        """
        if column not in self.df.columns:
            return f"Column '{column}' not found"
        
        fig = px.histogram(
            self.df,
            x=column,
            title = f"Histogram of {column}"
        )

        return fig

    def scatter_plot(self,col1,col2):
        """
        Generates interactive scatter plot between two columns.

        Args:
            col1(str): X-axis column
            col2(str): Y-axis column
        """

        if col1 not in self.df.columns or col2 not in self.df.columns:
            return "Invalid columns"

        fig = px.scatter(
            self.df,
            x = col1,
            y = col2,
            title = f"{col1} vs {col2}"
        )

        return fig
    
    def correlation_matrix(self):
        """
        Generates interactive correlation heatmap for numeric columns.
        """

        numeric_df = self.df.select_dtypes(include = np.number)

        if numeric_df.shape[1] == 0:
            return "No numeric columns available"

        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto = True,
            title = "Correlation Matrix"
        )

        return fig

    def label_encoding(self):
        """
        Applies label encoding to categorical columns.
        """
        if self.is_encoded:
            return self.df
        
        le = LabelEncoder()

        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = le.fit_transform(self.df[col].astype(str))
        
        self.is_encoded = True
        return self.df

    def one_hot_encoding(self):
        """
        Applies one-hot encoding to categorical columns.
        """
        return pd.get_dummies(self.df)

    def regression_model(self):
        """
        Builds a linear regression model and returns performance metrics.
        """
        df = self.label_encoding()
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for regression"

        X = df[numeric_cols[:-1]]
        y = df[numeric_cols[-1]]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        model = LinearRegression()
        model.fit(X_train,y_train)

        preds = model.predict(X_test)

        return{
            "r2_score": r2_score(y_test,preds),
            "mse": mean_squared_error(y_test,preds)
        }

    def classification_model(self):
        """
        Builds a classification model using Random Forest.
        """
        df = self.label_encoding()
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for classification"

        X = df[numeric_cols[:-1]]
        y = df[numeric_cols[-1]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds, average='weighted'),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }

    def clustering_model(self, n_clusters=3):
        """
        Builds a KMeans clustering model.

        Args:
            n_clusters (int): Number of clusters to form
        """
        df = self.label_encoding()
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) < 2:
            return "Not enough numeric columns for clustering"

        X = df[numeric_cols]

        model = KMeans(n_clusters=n_clusters, n_init = 10, random_state=42)
        labels = model.fit_predict(X)

        return {
            "clusters": labels.tolist()
        }

    def run_analysis(self,plan=None):
        """
        Executes analysis based on provided plan.
        If no plan is provided, runs default pipeline.
        """

        self.results = {}

        # default flow
        if plan is None:
            self.results['preview'] = self.preview_data()
            self.results['nulls'] = self.null_analysis()
            self.results['descripton'] = self.describe_data()
            return self.results

        # plan based execution
        for step in plan:

            action = step.get('step')

            if action == "preview_data":
                self.results['preview'] = self.preview_data()
            
            elif action == "null_analysis":
                self.results["nulls"] = self.null_analysis()

            elif action == "describe_data":
                self.results['description'] = self.describe_data()

            elif action == "histogram":
                col = step.get("column")
                if col in self.df.columns:
                    self.results.setdefault("plots",{}).setdefault("histograms",[]).append(
                        {"column": col, "figure": self.histogram(col)}
                    )
                else:
                    self.results.setdefault("plots",{})["histogram"] = f"{col} not found"

            elif action == "scatter_plot":
                col1 = step.get("col1")
                col2 = step.get("col2")

                if col1 in self.df.columns and col2 in self.df.columns:
                    self.results.setdefault("plots",{}).setdefault("scatter_plots",[]).append(
                        {"col1": col1, "col2": col2, "figure": self.scatter_plot(col1, col2)}
                    )
                else:
                    self.results.setdefault("plots",{})["scatter"] = "Invalid columns for scatter plot"

            elif action == "correlation_matrix":
                self.results.setdefault("plots",{})["correlation"] = self.correlation_matrix()

            elif action == "regression_model":
                self.results.setdefault("models",{})["regression"] = self.regression_model()

            elif action == "classification_model":
                self.results.setdefault("models",{})["classification"] = self.classification_model()

            elif action == "clustering_model":
                self.results.setdefault("models",{})["clustering"] = self.clustering_model()

        return self.results