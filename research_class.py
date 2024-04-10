# импорт модулей
# import os
# import phik
# import shap
# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from datetime import timedelta
# from autofeat import AutoFeatClassifier
# from phik.report import plot_correlation_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix
# from sklearn.model_selection import (
#     train_test_split,
#     StratifiedKFold,
#     cross_validate
# )

RANDOM_STATE = 42
# sns.set_style("white")
# sns.set_theme(style="whitegrid")

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

	# def data_preprocessing(self, dropnas=True, date_features=None, int_features=None, drop_features=None):
 #        # удаление дубликатов
	# 	self.dataset.drop_duplicates(inplace=True)
	# 	self.dataset.reset_index(drop=True, inplace=True)
 #        # удаление пропущенных значений
	# 	if dropnas:
	# 		self.dataset.dropna(inplace=True)
        
 #        # изменение типов данных для дат
	# 	if date_features is not None:
	# 		if isinstance(date_features, list):
	# 			rows_deleted = 0
	# 			for col in date_features:
	# 				try:
	# 					self.dataset[col] = pd.to_datetime(self.dataset[col], format='%Y-%m-%d %H:%M:%S.%f')
	# 				except pd.errors.OutOfBoundsDatetime as e:
	# 					error_message = str(e)
	# 					print(f"Ошибка при обработке даты в столбце {col}: {error_message}")
	# 					error_indices = self.dataset[self.dataset[col] == error_message].index
	# 					self.dataset.drop(index=error_indices, inplace=True)
	# 					rows_deleted += len(error_indices)
	# 			if rows_deleted > 0:
	# 				print(f"По причине некорректного значения даты удалено {rows_deleted} строк.")
	# 		else:
	# 			try:
	# 				self.dataset[date_features] = pd.to_datetime(self.dataset[date_features], format='%Y-%m-%d %H:%M:%S.%f')
	# 			except pd.errors.OutOfBoundsDatetime as e:
	# 				error_message = str(e)
	# 				print(f"Ошибка при обработке даты в столбце {date_features}: {error_message}")
	# 				error_indices = self.dataset[self.dataset[date_features] == error_message].index
	# 				self.dataset.drop(index=error_indices, inplace=True)
	# 				print(f"По причине некорректного значения даты удалено {len(error_indices)} строк.")
        
 #        # изменение типов данных для целочисленных значений
	# 	if int_features is not None:
	# 		if isinstance(int_features, list):
	# 			for col in int_features:
	# 				try:
	# 					self.dataset[col] = self.dataset[col].astype('int')
	# 				except pd.errors.IntCastingNaNError:
	# 					self.dataset[col] = self.dataset[col].fillna(-1).astype('int')
	# 		else:
	# 			try:
	# 				self.dataset[int_features] = self.dataset[int_features].astype('int')
	# 			except pd.errors.IntCastingNaNError:
	# 				self.dataset[int_features] = self.dataset[int_features].fillna(-1).astype('int')
        
 #        # удаление ненужных признаков, если drop_features не равен None
	# 	if drop_features is not None:
	# 		if isinstance(drop_features, list):
	# 			self.dataset.drop(columns=drop_features, axis=1, inplace=True)
        
 #        # отображение обновлённого датасета
	# 	self.dataset.info()
	# 	display(self.dataset.head())
	# 	return self.dataset

	# def exploratory_data_analysis(self, table_name=None, drop_columns=None, interval_cols=None, size=(8, 8), assets_dir=None):
	# 	phik_overview = self.dataset.drop(columns=drop_columns).phik_matrix(interval_cols=interval_cols)
	# 	sns.set()
	# 	plot_correlation_matrix(phik_overview.values,
	# 							x_labels=phik_overview.columns,
	# 							y_labels=phik_overview.index,
	# 							vmin=0,
	# 							vmax=1,
	# 							fontsize_factor=0.8,
	# 							figsize=size)
	# 	plt.xticks(rotation=45)
	# 	plt.title(f'Корреляция между признаками в таблице {table_name}', fontsize=12, y=1.02)
	# 	plt.tight_layout()
	# 	if assets_dir:
	# 		plt.savefig(os.path.join(assets_dir, f'target_count_{table_name}.png'))

	# def feature_engineering(self, test_size=0.2, features=None, target=None, add_features=False, transformations=None, categorical_features=None):
	# 	X_train, X_test, y_train, y_test = train_test_split(self.dataset[features],
	# 														self.dataset[target],
	# 														test_size=test_size,
	# 														random_state=RANDOM_STATE,
	# 														stratify=self.dataset[target])
	# 	scaler = StandardScaler()
	# 	X_train_scl = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
	# 	X_test_scl = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
	# 	if add_features:
	# 		transformations = transformations
	# 		afc = AutoFeatClassifier(categorical_cols=categorical_features, feateng_steps=1, max_gb=2, transformations=transformations, n_jobs=-1)
	# 		X_train_afc = afc.fit_transform(X_train, y_train)
	# 		X_test_afc = afc.transform(X_test)
	# 		print(X_train_afc.shape, X_test_afc.shape, y_train.shape, y_test.shape)
	# 		return X_train_afc, X_test_afc, y_train, y_test
	# 	else:
	# 		print(X_train_scl.shape, X_test_scl.shape, y_train.shape, y_test.shape)
	# 		return X_train_scl, X_test_scl, y_train, y_test

	# def feature_selection(self):
	# 	pass
	
	# def model_fitting(self, model_name=None, features_train=None, target_train=None, params=None):
	# 	if model_name == 'Baseline' or model_name == 'Logistic Regression':
	# 		model = LogisticRegression(**params)
	# 	elif model_name == 'Random Forest':
	# 		model = RandomForestClassifier(**params)
	# 	model.fit(features_train, target_train)
	# 	cv_strategy = StratifiedKFold(n_splits=4)
	# 	cv_res = cross_validate(model,
	# 							features_train,
	# 							target_train,
	# 							cv=cv_strategy,
	# 							n_jobs=-1,
	# 							scoring=['roc_auc', 'f1_micro', 'f1', 'f1_weighted', 'f1_macro'])
	# 	for key, value in cv_res.items():
	# 		cv_res[key] = round(value.mean(), 3)
	# 	print(f"результаты кросс-вадидации: {cv_res}")
	# 	y_pred = model.predict(features_train.values)
	# 	y_pred_proba = model.predict_proba(features_train.values)[:, 1]
				
	# 	roc_auc_value = roc_auc_score(target_train, y_pred_proba)
	# 	f1_value = f1_score(target_train, y_pred)
	# 	# Визуализация кривой ROC
	# 	fpr, tpr, thresholds = roc_curve(target_train, y_pred_proba)
	# 	sns.set_style('darkgrid')
	# 	plt.plot(fpr, tpr, linewidth=1.5, label='ROC-AUC (area = %0.2f)' % roc_auc_value)
	# 	plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label='random_classifier')
	# 	plt.xlim([-0.05, 1.0])
	# 	plt.ylim([0.0, 1.05])
	# 	plt.xlabel('False Positive Rate', fontsize=11)
	# 	plt.ylabel('True Positive Rate', fontsize=11)
	# 	plt.title('%s Receiver Operating Characteristic' % model_name, fontsize=12)
	# 	plt.legend(loc='lower right')
	# 	plt.show()

	# 	return cv_res['test_f1'], cv_res['test_roc_auc'], model

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
