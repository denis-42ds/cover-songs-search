# импорт модулей
import os
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from catboost import Pool, CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV

# установка констант
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
sns.set_style("white")
sns.set_theme(style="whitegrid")
pd.options.display.max_columns = 100
pd.options.display.max_rows = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetExplorer:
	def __init__(self, dataset, target=None):
		self.dataset = dataset
		self.target = target

	def explore_dataset(self, assets_dir=None, list_column=None):
		# Вывод информации о датасете
		self.dataset.info()

		# Вывод случайных примеров из датасета
		display(self.dataset.sample(5))

		if list_column:
			self.dataset[list_column] = self.dataset[list_column].apply(lambda x: str(x))

		# Количество полных дубликатов строк
		print(f"количество полных дубликатов строк: {self.dataset.duplicated().sum()}")

		# Круговая диаграмма для количества полных дубликатов
		if self.dataset.duplicated().sum() > 0:
			sizes = [self.dataset.duplicated().sum(), self.dataset.shape[0]]
			fig1, ax1 = plt.subplots()
			ax1.pie(sizes, labels=['duplicate', 'not a duplicate'], autopct='%1.0f%%')
			plt.title('Количество полных дубликатов в общем количестве строк', size=12)
			plt.show()

		print(f"""количество пропущенных значений:\n{self.dataset.isnull().sum()}""")
		if self.dataset.isnull().values.any():
			if self.dataset.shape[1] <= 20 or self.dataset.shape[0] < 800000:
				sns.heatmap(self.dataset.isnull(), cmap=sns.color_palette(['#000099', '#ffff00']))
				plt.xticks(rotation=90)
				plt.title('Визуализация количества пропущенных значений', size=12, y=1.02)
				plt.show()

			# Вывод признаков с пропущенными значениями
			missing_values_ratios = {}
			for column in self.dataset.columns[self.dataset.isna().any()].tolist():
				missing_values_ratio = self.dataset[column].isna().sum() / self.dataset.shape[0]
				missing_values_ratios[column] = missing_values_ratio

			print("Процент пропущенных значений в признаках:")
			for column, ratio in missing_values_ratios.items():
			    print(f"{column}: {ratio*100:.2f}%")

		# Выбираем только те признаки, у которых в названии есть 'id'
		id_columns = [col for col in self.dataset.columns if 'id' in col.lower()]

		# Выводим информацию для каждого выбранного признака
		for col in id_columns:
			print(f"Количество уникальных значений в столбце '{col}': {self.dataset[col].nunique()}")
			print(f"Соотношение уникальных значений и общего количества записей в столбце '{col}': {self.dataset[col].nunique() / self.dataset.shape[0]:.2f}")
			print()

		if self.target:
			print(f"""Соотношение классов целевой переменной:
			{self.dataset[self.target].value_counts().sort_index(ascending=False)}""")
			target_agg = self.dataset[self.target].value_counts().reset_index()
			sns.barplot(data=target_agg, x=self.target, y='count', hue=self.target, palette=sns.color_palette("husl", len(target_agg)))
			plt.title(f"{self.target} total distribution")
			if assets_dir:
				plt.savefig(os.path.join(assets_dir, 'target_count.png'))
			plt.show()

	def data_preprocessing(self, text_sentences=None, target_encoder='LabelEncoder', data_columns=None, cluster_model='KMeans', data_type='baseline_data', assets_dir=None):
		# получение эмбеддингов
		try:
			text_embeddings = pd.read_csv('data/text_embeddings.csv')
		except:
			model = SentenceTransformer('sentence-transformers/LaBSE')
			text_embeddings = model.encode(text_sentences)
			text_embeddings_df = pd.DataFrame(text_embeddings)
			text_embeddings_df.to_csv('data/text_embeddings.csv', index=False)
			print("Файл text_embeddings.csv успешно сохранен.")
		print(f"Размерность таблицы эмбеддингов: {text_embeddings.shape}")

		# кодирование целевой переменной
		if target_encoder == 'LabelEncoder':
			label_encoder = LabelEncoder()
			self.dataset['is_cover'] = label_encoder.fit_transform(np.array(self.dataset[self.target]).ravel())
			self.dataset['is_cover'] = self.dataset['is_cover'].map({0: 1, 1: 0})
			target = 'is_cover'
		else:
			print("input LabelEncoder")

		# разделение на обучающую и тестовую выборки и кластеризация
		new_data = text_embeddings.merge(self.dataset[data_columns], left_index=True, right_index=True)
		if data_type == 'baseline_data':
			X_train, X_test, y_train, y_test = train_test_split(new_data.drop(data_columns, axis=1),
				                                                new_data[target],
			                                                    test_size=0.1,
			                                                    random_state=RANDOM_STATE,
			                                                    stratify=new_data[target])
		else:
			if cluster_model == 'KMeans':
				cluster_model = KMeans(n_clusters=10, random_state=RANDOM_STATE, init='k-means++')
				cluster_labels = cluster_model.fit_predict(new_data)
				new_data['cluster'] = cluster_labels

			fig, axs = plt.subplots(2, 1)
			fig.tight_layout(pad=1.0)
			fig.set_size_inches(20, 10, forward=True)

			axs[0].scatter(new_data.iloc[:, 1], new_data.iloc[:, 0], c=cluster_labels, cmap='viridis', alpha=0.5)
			axs[0].scatter(cluster_model.cluster_centers_[:, 0], cluster_model.cluster_centers_[:, 1], marker='x', s=100, c='red', label='Centroids')
			axs[0].set_xlabel('Feature 1')
			axs[0].set_ylabel('Feature 2')
			axs[0].set_title('KMeans Clustering')
			axs[0].legend()

			sns.countplot(x='cluster', hue=target, data=new_data, ax=axs[1])
			axs[1].set_xlabel('Cluster_numbers')
			axs[1].set_ylabel('Count')
			axs[1].set_title('Распределение классов целевой переменной по кластерам')

			plt.show()

			centroids = cluster_model.cluster_centers_

			plt.figure(figsize=(20, 5))
			for i, centroid in enumerate(centroids):
				plt.subplot(1, len(centroids), i+1)
				plt.scatter(new_data.iloc[:, 0], new_data.iloc[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, edgecolors='k')
				plt.scatter(centroid[0], centroid[1], marker='x', s=200, c='red', label=f'Centroid {i}')
				plt.xlabel('Feature 1')
				plt.ylabel('Feature 2')
				plt.title(f'Centroid {i}')

			plt.suptitle('Диаграммы рассеяния для каждого центроида')
			plt.tight_layout()
			plt.show()

			plt.figure(figsize=(20, 5))
			for i, centroid in enumerate(centroids):
				plt.subplot(1, len(centroids), i+1)
				cluster_data = new_data[cluster_labels == i]
				plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c='blue', alpha=0.7, edgecolors='k')
				plt.scatter(centroid[0], centroid[1], marker='x', s=200, c='red', label=f'Centroid {i}')
				plt.xlabel('Feature 1')
				plt.ylabel('Feature 2')
				plt.title(f'Cluster {i}')

			plt.suptitle('Диаграммы рассеяния для каждого кластера и его центроида')
			plt.tight_layout()
			plt.show()

			X_train, X_test, y_train, y_test = train_test_split(new_data.drop(target, axis=1),
			                                                    new_data[target],
			                                                    test_size=0.1,
			                                                    random_state=RANDOM_STATE,
			                                                    stratify=new_data[target])
		
		print(f"Размерности полученных выборок: {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")

		return target, X_train, X_test, y_train, y_test

	def model_fitting(self, model_name=None, features_train=None, target_train=None, params=None, params_search=False, assets_dir=None):
		if params_search:
			pass
		else:
			if model_name == 'Baseline' or model_name == 'Logistic Regression':
				model = LogisticRegression(**params)
			elif model_name == 'CatBoost':
				model = CatBoostClassifier(**params)
			model.fit(features_train, target_train)
			cv_strategy = StratifiedKFold(n_splits=4)
			cv_res = cross_validate(model,
									features_train,
									target_train,
									cv=cv_strategy,
									n_jobs=-1,
									scoring=['roc_auc', 'f1'])
			for key, value in cv_res.items():
				cv_res[key] = round(value.mean(), 3)
			print(f"Результаты кросс-вадидации: {cv_res}")
				
		y_pred_proba = model.predict_proba(features_train)[:, 1]
		roc_auc_value = roc_auc_score(target_train, y_pred_proba)
		fpr, tpr, thresholds = roc_curve(target_train, y_pred_proba)
		plt.plot(fpr, tpr, linewidth=1.5, label='ROC-AUC (area = %0.2f)' % roc_auc_value)
		plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label='random_classifier')
		plt.xlim([-0.05, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate', fontsize=11)
		plt.ylabel('True Positive Rate', fontsize=11)
		plt.title(f"{model_name} Receiver Operating Characteristic", fontsize=12)
		plt.legend(loc='lower right')
		if assets_dir:
			plt.savefig(os.path.join(assets_dir, f'{model_name} Receiver Operating Characteristic.png'))
		plt.show()

		return cv_res, model

	# def test_best_model(self, model, features_train, features_test, target_test):
	# 	y_pred_proba = model.predict_proba(features_test.values)[:, 1]
	# 	roc_auc_value = roc_auc_score(target_test, y_pred_proba)
	# 	y_pred = model.predict(features_test.values)
	# 	f1_value = f1_score(target_test, y_pred)
        
	# 	print(f"ROC-AUC на тестовой выборке: {round(roc_auc_value, 2)}")
	# 	print(f"F1 на тестовой выборке: {round(f1_value, 2)}")

	# 	fig, axs = plt.subplots(1, 2)
	# 	fig.tight_layout(pad=1.0)
	# 	fig.set_size_inches(18, 6, forward=True)
		
	# 	sns.heatmap(confusion_matrix(target_test, y_pred.round()), annot=True, fmt='3.0f', cmap='crest', ax=axs[0])
	# 	axs[0].set_title('Матрица ошибок', fontsize=16, y=1.02)
		
	# 	features_importance = pd.DataFrame(data = {'feature': features_train.columns, 'percent': np.round(model.feature_importances_, decimals=1)})
	# 	axs[1].bar(features_importance.sort_values('percent', ascending=False)['feature'][:5], features_importance.sort_values('percent', ascending=False)['percent'][:5])
	# 	axs[1].set_xticks(range(5))
	# 	axs[1].set_xticklabels(features_importance['feature'][:5].unique(), rotation=45)
	# 	axs[1].set_title("Важность признаков", fontsize=16, y=1.02)
	# 	plt.show()
