"""
Advanced Machine Learning Engine for AI Data Scientist
Provides comprehensive ML capabilities: Regression, Classification, Time Series,
Feature Engineering, Dimensionality Reduction, NLP, and AutoML.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score
)

# Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
warnings.filterwarnings('ignore')


class MLEngine:
    """Core Machine Learning Engine with comprehensive capabilities."""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scaler = None
        self.label_encoder = None

    # ==================== REGRESSION MODELS ====================

    def get_regression_models(self):
        """Return dictionary of available regression models."""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'KNN Regressor': KNeighborsRegressor(n_neighbors=5)
        }

    def train_regression(self, X, y, model_name='Random Forest', test_size=0.2):
        """
        Train a regression model.

        Returns:
            dict with model, predictions, metrics, and feature importance
        """
        models = self.get_regression_models()
        if model_name not in models:
            return {"error": f"Unknown model: {model_name}"}

        # Handle missing values
        X = X.copy()
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'missing')

        # Encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        model = models[model_name]
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Metrics
        metrics = {
            'train_r2': round(r2_score(y_train, y_pred_train), 4),
            'test_r2': round(r2_score(y_test, y_pred_test), 4),
            'train_rmse': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4),
            'test_rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
            'train_mae': round(mean_absolute_error(y_train, y_pred_train), 4),
            'test_mae': round(mean_absolute_error(y_test, y_pred_test), 4),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = round(cv_scores.mean(), 4)
        metrics['cv_r2_std'] = round(cv_scores.std(), 4)

        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_encoded.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X_encoded.columns, np.abs(model.coef_)))

        self.models[model_name] = model

        return {
            'model': model,
            'model_name': model_name,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': {
                'y_train': y_train.values,
                'y_test': y_test.values,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            },
            'feature_names': X_encoded.columns.tolist(),
            'X_test': X_test,
            'scaler': self.scaler
        }

    # ==================== CLASSIFICATION MODELS ====================

    def get_classification_models(self):
        """Return dictionary of available classification models."""
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }

    def train_classification(self, X, y, model_name='Random Forest', test_size=0.2):
        """
        Train a classification model.

        Returns:
            dict with model, predictions, metrics, confusion matrix, and feature importance
        """
        models = self.get_classification_models()
        if model_name not in models:
            return {"error": f"Unknown model: {model_name}"}

        # Handle missing values
        X = X.copy()
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'missing')

        # Encode categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Encode target if needed
        self.label_encoder = LabelEncoder()
        if not pd.api.types.is_numeric_dtype(y):
            y_encoded = self.label_encoder.fit_transform(y)
            class_names = self.label_encoder.classes_
        else:
            y_encoded = y.values if hasattr(y, 'values') else y
            class_names = np.unique(y_encoded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        model = models[model_name]
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Probabilities (for ROC curve)
        y_prob_test = None
        if hasattr(model, 'predict_proba'):
            y_prob_test = model.predict_proba(X_test_scaled)

        # Metrics
        is_binary = len(np.unique(y_encoded)) == 2
        avg_method = 'binary' if is_binary else 'weighted'

        metrics = {
            'train_accuracy': round(accuracy_score(y_train, y_pred_train), 4),
            'test_accuracy': round(accuracy_score(y_test, y_pred_test), 4),
            'precision': round(precision_score(y_test, y_pred_test, average=avg_method, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred_test, average=avg_method, zero_division=0), 4),
            'f1': round(f1_score(y_test, y_pred_test, average=avg_method, zero_division=0), 4),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        metrics['cv_accuracy_mean'] = round(cv_scores.mean(), 4)
        metrics['cv_accuracy_std'] = round(cv_scores.std(), 4)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)

        # Classification Report
        class_report = classification_report(y_test, y_pred_test, output_dict=True)

        # ROC Curve (for binary classification)
        roc_data = None
        if is_binary and y_prob_test is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob_test[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': round(roc_auc, 4)}
            metrics['auc'] = round(roc_auc, 4)

        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_encoded.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            feature_importance = dict(zip(X_encoded.columns, np.abs(coef)))

        self.models[model_name] = model

        return {
            'model': model,
            'model_name': model_name,
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'roc_data': roc_data,
            'feature_importance': feature_importance,
            'predictions': {
                'y_train': y_train,
                'y_test': y_test,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'y_prob_test': y_prob_test
            },
            'class_names': class_names,
            'feature_names': X_encoded.columns.tolist(),
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }

    # ==================== AUTOML ====================

    def auto_ml(self, X, y, task='auto', test_size=0.2, top_n=3):
        """
        Automatically train and compare multiple models.

        Args:
            X: Features DataFrame
            y: Target Series
            task: 'classification', 'regression', or 'auto' (detect from target)
            top_n: Number of top models to return

        Returns:
            dict with comparison results and best model
        """
        # Auto-detect task type
        if task == 'auto':
            unique_ratio = len(y.unique()) / len(y)
            if pd.api.types.is_numeric_dtype(y) and unique_ratio > 0.1:
                task = 'regression'
            else:
                task = 'classification'

        results = []

        if task == 'regression':
            models = self.get_regression_models()
            for name in models.keys():
                try:
                    result = self.train_regression(X, y, model_name=name, test_size=test_size)
                    if 'error' not in result:
                        results.append({
                            'model_name': name,
                            'test_r2': result['metrics']['test_r2'],
                            'test_rmse': result['metrics']['test_rmse'],
                            'cv_r2': result['metrics']['cv_r2_mean'],
                            'full_result': result
                        })
                except Exception as e:
                    continue

            # Sort by test R2
            results = sorted(results, key=lambda x: x['test_r2'], reverse=True)
            metric_name = 'test_r2'

        else:  # classification
            models = self.get_classification_models()
            for name in models.keys():
                try:
                    result = self.train_classification(X, y, model_name=name, test_size=test_size)
                    if 'error' not in result:
                        results.append({
                            'model_name': name,
                            'test_accuracy': result['metrics']['test_accuracy'],
                            'f1': result['metrics']['f1'],
                            'cv_accuracy': result['metrics']['cv_accuracy_mean'],
                            'full_result': result
                        })
                except Exception as e:
                    continue

            # Sort by test accuracy
            results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
            metric_name = 'test_accuracy'

        return {
            'task': task,
            'all_results': results,
            'top_models': results[:top_n],
            'best_model': results[0] if results else None,
            'comparison_metric': metric_name
        }

    # ==================== FEATURE ENGINEERING ====================

    def create_polynomial_features(self, df, columns, degree=2, include_bias=False):
        """Create polynomial features from specified columns."""
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X = df[columns].values
        X_poly = poly.fit_transform(X)

        feature_names = poly.get_feature_names_out(columns)
        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)

        return pd.concat([df, df_poly.drop(columns, axis=1)], axis=1)

    def create_interaction_features(self, df, columns):
        """Create interaction features between specified columns."""
        df_result = df.copy()

        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    df_result[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    df_result[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)

        return df_result

    def create_datetime_features(self, df, column):
        """Extract datetime features from a datetime column."""
        df_result = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            df_result[column] = pd.to_datetime(df[column], errors='coerce')

        dt_col = df_result[column]

        df_result[f'{column}_year'] = dt_col.dt.year
        df_result[f'{column}_month'] = dt_col.dt.month
        df_result[f'{column}_day'] = dt_col.dt.day
        df_result[f'{column}_dayofweek'] = dt_col.dt.dayofweek
        df_result[f'{column}_quarter'] = dt_col.dt.quarter
        df_result[f'{column}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
        df_result[f'{column}_hour'] = dt_col.dt.hour

        return df_result

    def create_binned_features(self, df, column, n_bins=5, strategy='quantile'):
        """Create binned features from a numeric column."""
        df_result = df.copy()

        if strategy == 'quantile':
            df_result[f'{column}_binned'] = pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
        else:
            df_result[f'{column}_binned'] = pd.cut(df[column], bins=n_bins, labels=False)

        return df_result

    def select_features(self, X, y, method='mutual_info', k=10):
        """
        Feature selection using various methods.

        Args:
            method: 'mutual_info', 'f_score', 'rfe'
            k: number of features to select
        """
        # Handle categorical columns
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Handle missing values
        X_encoded = X_encoded.fillna(X_encoded.median())

        # Determine if classification or regression
        is_classification = len(y.unique()) < 20

        if method == 'mutual_info':
            if is_classification:
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X_encoded.columns)))
            else:
                from sklearn.feature_selection import mutual_info_regression
                selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(X_encoded.columns)))
        elif method == 'f_score':
            if is_classification:
                selector = SelectKBest(score_func=f_classif, k=min(k, len(X_encoded.columns)))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(k, len(X_encoded.columns)))
        elif method == 'rfe':
            if is_classification:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, len(X_encoded.columns)))

        # Handle target encoding
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y

        selector.fit(X_encoded, y_encoded)

        if method == 'rfe':
            scores = selector.ranking_
            selected_mask = selector.support_
        else:
            scores = selector.scores_
            selected_mask = selector.get_support()

        feature_scores = dict(zip(X_encoded.columns, scores))
        selected_features = X_encoded.columns[selected_mask].tolist()

        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'method': method
        }

    # ==================== DIMENSIONALITY REDUCTION ====================

    def perform_pca(self, df, columns, n_components=2, return_explained_variance=True):
        """
        Perform Principal Component Analysis.
        """
        X = df[columns].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Create result DataFrame
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

        result = {
            'pca_data': df_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pd.DataFrame(
                pca.components_,
                columns=columns,
                index=pca_cols
            ),
            'n_components': n_components
        }

        return result

    def perform_tsne(self, df, columns, n_components=2, perplexity=30, n_iter=1000):
        """
        Perform t-SNE dimensionality reduction.
        """
        X = df[columns].dropna()

        # Sample if too large
        if len(X) > 5000:
            X = X.sample(5000, random_state=42)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(X)-1),
                        n_iter=n_iter, random_state=42)
        except TypeError:
            tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(X)-1),
                        max_iter=n_iter, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)

        tsne_cols = [f'tSNE{i+1}' for i in range(n_components)]
        df_tsne = pd.DataFrame(X_tsne, columns=tsne_cols, index=X.index)

        return {
            'tsne_data': df_tsne,
            'original_indices': X.index.tolist()
        }

    # ==================== TIME SERIES ANALYSIS ====================

    def analyze_time_series(self, df, date_column, value_column, freq='D'):
        """
        Comprehensive time series analysis.
        """
        df_ts = df[[date_column, value_column]].copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        df_ts = df_ts.sort_values(date_column)
        df_ts = df_ts.set_index(date_column)

        # Resample if needed
        if freq:
            df_ts = df_ts.resample(freq).mean().dropna()

        series = df_ts[value_column]

        results = {
            'series': series,
            'basic_stats': {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'trend': 'upward' if series.iloc[-1] > series.iloc[0] else 'downward'
            }
        }

        # Stationarity test (ADF)
        try:
            adf_result = adfuller(series.dropna())
            results['stationarity'] = {
                'adf_statistic': round(adf_result[0], 4),
                'p_value': round(adf_result[1], 4),
                'is_stationary': adf_result[1] < 0.05
            }
        except:
            results['stationarity'] = None

        # Seasonal decomposition
        try:
            if len(series) >= 14:  # Need enough data points
                decomposition = seasonal_decompose(series, model='additive', period=min(7, len(series)//2))
                results['decomposition'] = {
                    'trend': decomposition.trend,
                    'seasonal': decomposition.seasonal,
                    'residual': decomposition.resid
                }
        except:
            results['decomposition'] = None

        # ACF and PACF
        try:
            nlags = min(20, len(series) // 2 - 1)
            if nlags > 0:
                results['acf'] = acf(series.dropna(), nlags=nlags)
                results['pacf'] = pacf(series.dropna(), nlags=nlags)
        except:
            pass

        return results

    def forecast_time_series(self, df, date_column, value_column, periods=30, method='auto'):
        """
        Time series forecasting.

        Args:
            method: 'arima', 'exponential_smoothing', or 'auto'
        """
        df_ts = df[[date_column, value_column]].copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        df_ts = df_ts.sort_values(date_column)
        df_ts = df_ts.set_index(date_column)
        df_ts = df_ts.asfreq('D')  # Daily frequency
        df_ts[value_column] = df_ts[value_column].interpolate()

        series = df_ts[value_column].dropna()

        # Split for validation
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        results = {'method': method, 'periods': periods}

        try:
            if method in ['arima', 'auto']:
                # Simple ARIMA or Auto-ARIMA
                if method == 'arima': # Force simple
                     model = ARIMA(train, order=(1, 1, 1))
                     fitted = model.fit()
                else: # Auto (try to optimize)
                     # Use a smaller subset for grid search if data is large to save time
                     search_train = train if len(train) < 200 else train[-200:]
                     fitted, best_order = self.auto_arima_search(search_train)
                     # Re-fit on full train with best order
                     if fitted:
                         model = ARIMA(train, order=best_order)
                         fitted = model.fit()
                     else:
                         model = ARIMA(train, order=(1,1,1))
                         fitted = model.fit()

                # Forecast
                forecast = fitted.forecast(steps=periods)

                # Validation metrics
                val_forecast = fitted.forecast(steps=len(test))
                # Handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((test.values - val_forecast.values) / test.values)) * 100
                    if np.isinf(mape) or np.isnan(mape): mape = 0

                results['arima'] = {
                    'forecast': forecast,
                    'validation_mape': round(mape, 2),
                    'aic': round(fitted.aic, 2)
                }

            if method in ['exponential_smoothing', 'auto']:
                # Exponential Smoothing
                model = ExponentialSmoothing(train, trend='add', seasonal=None)
                fitted = model.fit()

                forecast = fitted.forecast(steps=periods)

                val_forecast = fitted.forecast(steps=len(test))
                mape = np.mean(np.abs((test.values - val_forecast.values) / test.values)) * 100

                results['exp_smoothing'] = {
                    'forecast': forecast,
                    'validation_mape': round(mape, 2)
                }

            # Choose best method
            if method == 'auto' and 'arima' in results and 'exp_smoothing' in results:
                if results['arima']['validation_mape'] < results['exp_smoothing']['validation_mape']:
                    results['best_method'] = 'arima'
                    results['best_forecast'] = results['arima']['forecast']
                else:
                    results['best_method'] = 'exponential_smoothing'
                    results['best_forecast'] = results['exp_smoothing']['forecast']
            elif 'arima' in results:
                results['best_method'] = 'arima'
                results['best_forecast'] = results['arima']['forecast']
            elif 'exp_smoothing' in results:
                results['best_method'] = 'exponential_smoothing'
                results['best_forecast'] = results['exp_smoothing']['forecast']

        except Exception as e:
            results['error'] = str(e)

        return results

    # ==================== NLP / TEXT ANALYSIS ====================

    def analyze_text(self, df, column):
        """
        Basic text analysis without external NLP libraries.
        """
        texts = df[column].dropna().astype(str)

        # Basic statistics
        word_counts = texts.apply(lambda x: len(x.split()))
        char_counts = texts.apply(len)

        # Simple word frequency
        all_words = ' '.join(texts).lower().split()
        word_freq = pd.Series(all_words).value_counts()

        # Sentiment approximation (very basic)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                         'love', 'best', 'happy', 'awesome', 'perfect', 'beautiful'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst',
                         'poor', 'disappointing', 'sad', 'angry', 'ugly', 'wrong'}

        def simple_sentiment(text):
            words = set(text.lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            if pos > neg:
                return 'positive'
            elif neg > pos:
                return 'negative'
            return 'neutral'

        sentiments = texts.apply(simple_sentiment)

        return {
            'total_documents': len(texts),
            'avg_word_count': round(word_counts.mean(), 2),
            'avg_char_count': round(char_counts.mean(), 2),
            'top_words': word_freq.head(20).to_dict(),
            'sentiment_distribution': sentiments.value_counts().to_dict(),
            'word_count_stats': {
                'min': int(word_counts.min()),
                'max': int(word_counts.max()),
                'median': int(word_counts.median())
            }
        }

    def create_text_features(self, df, column):
        """Create numeric features from text column."""
        df_result = df.copy()
        texts = df[column].fillna('').astype(str)

        df_result[f'{column}_word_count'] = texts.apply(lambda x: len(x.split()))
        df_result[f'{column}_char_count'] = texts.apply(len)
        df_result[f'{column}_avg_word_length'] = texts.apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
        )
        df_result[f'{column}_uppercase_ratio'] = texts.apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )
        df_result[f'{column}_digit_ratio'] = texts.apply(
            lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1)
        )

        return df_result

    # ==================== ANOMALY DETECTION ====================

    def detect_anomalies(self, df, columns, method='isolation_forest', contamination=0.1):
        """
        Detect anomalies using various methods.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        X = df[columns].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
            predictions = detector.fit_predict(X_scaled)
        elif method == 'lof':
            detector = LocalOutlierFactor(contamination=contamination)
            predictions = detector.fit_predict(X_scaled)

        # -1 for anomalies, 1 for normal
        anomaly_mask = predictions == -1

        df_result = df.copy()
        df_result['is_anomaly'] = False
        df_result.loc[X.index, 'is_anomaly'] = anomaly_mask

        return {
            'data_with_anomalies': df_result,
            'anomaly_count': int(anomaly_mask.sum()),
            'anomaly_percentage': round(anomaly_mask.sum() / len(X) * 100, 2),
            'anomaly_indices': X.index[anomaly_mask].tolist()
        }


# Singleton instance

    def optimize_hyperparameters(self, X, y, model_name, task='classification', cv=3):
        """
        Perform Grid Search for hyperparameter tuning.
        """
        from sklearn.model_selection import GridSearchCV

        # Define grids for common models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            },
             'Linear Regression': {
                'fit_intercept': [True, False]
            },
             'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0]
             },
             'Lasso Regression': {
                'alpha': [0.1, 1.0, 10.0]
             },
             'Decision Tree': {
                 'max_depth': [None, 10, 20, 30],
                 'min_samples_split': [2, 5, 10]
             },
             'Gradient Boosting': {
                 'n_estimators': [50, 100, 200],
                 'learning_rate': [0.01, 0.1, 0.2],
                 'max_depth': [3, 5, 7]
             },
             'SVM': {
                 'C': [0.1, 1, 10],
                 'kernel': ['rbf', 'linear']
             },
             'KNN': {
                 'n_neighbors': [3, 5, 7, 9],
                 'weights': ['uniform', 'distance']
             }
        }

        # Get base model
        if task == 'regression':
            base_model = self.get_regression_models().get(model_name)
        else:
            base_model = self.get_classification_models().get(model_name)

        if not base_model:
            return {'error': f"Unknown model: {model_name}"}

        # Get params
        params = param_grids.get(model_name)
        if not params:
             return {'error': f"No hyperparameter grid defined for {model_name}"}

        # Preprocessing
        X = X.copy()
        for col in X.columns:
            if X[col].isnull().any():
                 X[col] = X[col].fillna(0) # Simple fill for speed

        X_encoded = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        if task != 'regression' and not pd.api.types.is_numeric_dtype(y):
             le = LabelEncoder()
             y = le.fit_transform(y)

        # Run Grid Search
        grid = GridSearchCV(base_model, params, cv=cv, scoring='r2' if task == 'regression' else 'accuracy', n_jobs=-1)
        grid.fit(X_scaled, y)

        return {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_,
            'best_estimator': grid.best_estimator_
        }

    def auto_arima_search(self, series, max_p=3, max_d=2, max_q=3):
        """
        Simple grid search for ARIMA parameters.
        """
        best_aic = float('inf')
        best_order = (1, 1, 1)
        best_model = None

        import itertools
        p = range(0, max_p)
        d = range(0, max_d)
        q = range(0, max_q)

        pdq = list(itertools.product(p, d, q))

        for param in pdq:
            try:
                model = ARIMA(series, order=param)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
                    best_model = results
            except:
                continue

        return best_model, best_order
ml_engine = MLEngine()
