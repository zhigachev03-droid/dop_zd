#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единое приложение для анализа мошеннических транзакций
Объединяет анализ данных, визуализацию и машинное обучение
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
# Устанавливаем backend для PyQt6
matplotlib.use('QtAgg')
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QFileDialog, QTextEdit, QProgressBar, QGroupBox,
                             QGridLayout, QSplitter, QFrame, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

warnings.filterwarnings('ignore')

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

class DataLoader:
    """Класс для загрузки и подготовки данных"""
    
    def __init__(self):
        self.transactions_df = None
        self.currency_df = None
        self.is_data_loaded = False
        
    def load_transactions(self, file_path):
        """Загрузка данных о транзакциях"""
        try:
            self.transactions_df = pd.read_parquet(file_path)
            print(f"Загружено {len(self.transactions_df):,} транзакций")
            return True
        except Exception as e:
            print(f"Ошибка загрузки файла транзакций: {e}")
            return False
            
    def load_currency(self, file_path):
        """Загрузка данных о курсах валют"""
        try:
            self.currency_df = pd.read_parquet(file_path)
            print(f"Загружено {len(self.currency_df)} записей о курсах валют")
            return True
        except Exception as e:
            print(f"Ошибка загрузки файла валют: {e}")
            return False
            
    def check_data_ready(self):
        """Проверка готовности данных"""
        self.is_data_loaded = (self.transactions_df is not None and self.currency_df is not None)
        return self.is_data_loaded

class AnalysisEngine:
    """Класс для выполнения анализа данных"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        # Кэш для результатов анализа
        self.analysis_cache = {}
        
    def clear_cache(self):
        """Очистка кэша анализа"""
        self.analysis_cache.clear()
        
    def _cache_result(self, method_name, result):
        """Кэширование результата анализа"""
        self.analysis_cache[method_name] = result
        return result
        
    def basic_data_overview(self):
        """Базовый обзор данных"""
        df = self.data_loader.transactions_df
        results = []
        
        results.append("="*60)
        results.append("БАЗОВЫЙ ОБЗОР ДАННЫХ")
        results.append("="*60)
        
        results.append(f"Размер датасета: {df.shape}")
        results.append(f"Количество транзакций: {len(df):,}")
        results.append(f"Количество признаков: {len(df.columns)}")
        
        results.append("\nКолонки:")
        for i, col in enumerate(df.columns, 1):
            results.append(f"{i:2d}. {col}")
        
        results.append("\nТипы данных:")
        results.append(str(df.dtypes))
        
        results.append("\nПропущенные значения:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Количество': missing_data,
            'Процент': missing_percent
        })
        missing_results = missing_df[missing_df['Количество'] > 0]
        if not missing_results.empty:
            results.append(str(missing_results))
        else:
            results.append("Пропущенных значений нет")
            
        return "\n".join(results)
        
    def fraud_analysis(self):
        """Анализ мошеннических транзакций"""
        # Проверяем кэш
        if 'fraud_analysis' in self.analysis_cache:
            return self.analysis_cache['fraud_analysis']
            
        df = self.data_loader.transactions_df
        results = []
        
        results.append("="*60)
        results.append("АНАЛИЗ МОШЕННИЧЕСКИХ ТРАНЗАКЦИЙ")
        results.append("="*60)
        
        # Распределение мошенничества
        fraud_counts = df['is_fraud'].value_counts()
        fraud_percent = (fraud_counts[True] / len(df)) * 100
        
        results.append(f"Всего транзакций: {len(df):,}")
        results.append(f"Легитимных транзакций: {fraud_counts[False]:,} ({100-fraud_percent:.1f}%)")
        results.append(f"Мошеннических транзакций: {fraud_counts[True]:,} ({fraud_percent:.1f}%)")
        
        # Анализ по категориям вендоров
        results.append("\nРаспределение мошенничества по категориям вендоров:")
        vendor_fraud = df.groupby('vendor_category')['is_fraud'].agg(['count', 'sum', 'mean'])
        vendor_fraud.columns = ['Всего транзакций', 'Мошеннических', 'Доля мошенничества']
        vendor_fraud = vendor_fraud.sort_values('Доля мошенничества', ascending=False)
        results.append(str(vendor_fraud))
        
        # Анализ по типам карт
        results.append("\nРаспределение мошенничества по типам карт:")
        card_fraud = df.groupby('card_type')['is_fraud'].agg(['count', 'sum', 'mean'])
        card_fraud.columns = ['Всего транзакций', 'Мошеннических', 'Доля мошенничества']
        card_fraud = card_fraud.sort_values('Доля мошенничества', ascending=False)
        results.append(str(card_fraud.head(10)))
        
        # Сохраняем структурированные данные для графиков
        fraud_data = {
            'text_result': "\n".join(results),
            'fraud_counts': fraud_counts,
            'vendor_fraud': vendor_fraud,
            'card_fraud': card_fraud,
            'channel_fraud': df.groupby('channel')['is_fraud'].mean().sort_values(ascending=False)
        }
        
        return self._cache_result('fraud_analysis', fraud_data)
        
    def temporal_analysis(self):
        """Анализ временных паттернов"""
        # Проверяем кэш
        if 'temporal_analysis' in self.analysis_cache:
            return self.analysis_cache['temporal_analysis']
            
        df = self.data_loader.transactions_df
        results = []
        
        results.append("="*60)
        results.append("ВРЕМЕННЫЙ АНАЛИЗ")
        results.append("="*60)
        
        # Конвертируем timestamp в datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Анализ по часам
        hourly_fraud = df.groupby('hour')['is_fraud'].mean()
        results.append("\nДоля мошенничества по часам дня:")
        for hour, fraud_rate in hourly_fraud.items():
            results.append(f"  {hour:02d}:00 - {fraud_rate:.3f}")
        
        # Анализ по дням недели
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_ru = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
        daily_fraud = df.groupby('day_of_week')['is_fraud'].mean().reindex(day_order)
        results.append("\nДоля мошенничества по дням недели:")
        for day, fraud_rate in zip(day_names_ru, daily_fraud.values):
            results.append(f"  {day} - {fraud_rate:.3f}")
        
        # Сохраняем структурированные данные для графиков
        temporal_data = {
            'text_result': "\n".join(results),
            'hourly_fraud': hourly_fraud,
            'daily_fraud': daily_fraud,
            'hourly_volume': df.groupby('hour').size(),
            'df_with_time': df  # DataFrame с добавленными временными колонками
        }
            
        return self._cache_result('temporal_analysis', temporal_data)
        
    def geographical_analysis(self):
        """Географический анализ"""
        df = self.data_loader.transactions_df
        results = []
        
        results.append("="*60)
        results.append("ГЕОГРАФИЧЕСКИЙ АНАЛИЗ")
        results.append("="*60)
        
        # Анализ по странам
        country_fraud = df.groupby('country')['is_fraud'].agg(['count', 'sum', 'mean'])
        country_fraud.columns = ['Всего транзакций', 'Мошеннических', 'Доля мошенничества']
        country_fraud = country_fraud.sort_values('Доля мошенничества', ascending=False)
        results.append("\nТоп-10 стран по количеству мошеннических транзакций:")
        results.append(str(country_fraud.head(10)))
        
        # Анализ по городам
        city_fraud = df.groupby('city')['is_fraud'].agg(['count', 'sum', 'mean'])
        city_fraud.columns = ['Всего транзакций', 'Мошеннических', 'Доля мошенничества']
        city_fraud = city_fraud.sort_values('Доля мошенничества', ascending=False)
        results.append("\nТоп-10 городов по количеству мошеннических транзакций:")
        results.append(str(city_fraud.head(10)))
        
        # Анализ транзакций вне страны проживания
        results.append(f"\nТранзакции вне страны проживания:")
        outside_home = df['is_outside_home_country'].value_counts()
        outside_home_fraud = df.groupby('is_outside_home_country')['is_fraud'].mean()
        
        for is_outside in [False, True]:
            count = outside_home[is_outside]
            fraud_rate = outside_home_fraud[is_outside]
            status = "вне страны" if is_outside else "в стране проживания"
            results.append(f"  {status}: {count:,} транзакций, доля мошенничества: {fraud_rate:.3f}")
        
        return "\n".join(results)
        
    def amount_analysis(self):
        """Анализ сумм транзакций"""
        df = self.data_loader.transactions_df
        results = []
        
        results.append("="*60)
        results.append("АНАЛИЗ СУММ ТРАНЗАКЦИЙ")
        results.append("="*60)
        
        # Общая статистика по суммам
        results.append("Статистика по суммам транзакций:")
        results.append(f"  Минимальная сумма: {df['amount'].min():.2f}")
        results.append(f"  Максимальная сумма: {df['amount'].max():.2f}")
        results.append(f"  Средняя сумма: {df['amount'].mean():.2f}")
        results.append(f"  Медианная сумма: {df['amount'].median():.2f}")
        
        # Анализ по типам транзакций
        results.append("\nСредние суммы по типам транзакций:")
        amount_by_fraud = df.groupby('is_fraud')['amount'].agg(['mean', 'median', 'std'])
        amount_by_fraud.columns = ['Средняя сумма', 'Медианная сумма', 'Стандартное отклонение']
        results.append(str(amount_by_fraud))
        
        # Анализ по категориям вендоров
        results.append("\nСредние суммы по категориям вендоров:")
        vendor_amounts = df.groupby('vendor_category')['amount'].agg(['mean', 'median', 'count'])
        vendor_amounts.columns = ['Средняя сумма', 'Медианная сумма', 'Количество транзакций']
        vendor_amounts = vendor_amounts.sort_values('Средняя сумма', ascending=False)
        results.append(str(vendor_amounts))
        
        return "\n".join(results)
        
    def device_channel_analysis(self):
        """Анализ устройств и каналов"""
        df = self.data_loader.transactions_df
        results = []
        
        results.append("="*60)
        results.append("АНАЛИЗ УСТРОЙСТВ И КАНАЛОВ")
        results.append("="*60)
        
        # Анализ по каналам
        results.append("Распределение мошенничества по каналам:")
        channel_fraud = df.groupby('channel')['is_fraud'].agg(['count', 'sum', 'mean'])
        channel_fraud.columns = ['Всего транзакций', 'Мошеннических', 'Доля мошенничества']
        channel_fraud = channel_fraud.sort_values('Доля мошенничества', ascending=False)
        results.append(str(channel_fraud))
        
        # Анализ по устройствам
        results.append("\nТоп-10 устройств по доле мошенничества:")
        device_fraud = df.groupby('device')['is_fraud'].agg(['count', 'sum', 'mean'])
        device_fraud.columns = ['Всего транзакций', 'Мошеннических', 'Доля мошенничества']
        device_fraud = device_fraud[device_fraud['Всего транзакций'] >= 100]  # Фильтруем редкие устройства
        device_fraud = device_fraud.sort_values('Доля мошенничества', ascending=False)
        results.append(str(device_fraud.head(10)))
        
        # Анализ присутствия карты
        results.append(f"\nТранзакции с присутствием карты:")
        card_present = df['is_card_present'].value_counts()
        card_present_fraud = df.groupby('is_card_present')['is_fraud'].mean()
        
        for is_present in [False, True]:
            count = card_present[is_present]
            fraud_rate = card_present_fraud[is_present]
            status = "с картой" if is_present else "без карты"
            results.append(f"  {status}: {count:,} транзакций, доля мошенничества: {fraud_rate:.3f}")
        
        return "\n".join(results)
        
    def last_hour_activity_analysis(self):
        """Анализ активности за последний час"""
        df = self.data_loader.transactions_df
        results = []
        
        results.append("="*60)
        results.append("АНАЛИЗ АКТИВНОСТИ ЗА ПОСЛЕДНИЙ ЧАС")
        results.append("="*60)
        
        # Извлекаем данные из структуры last_hour_activity
        if 'last_hour_activity' in df.columns:
            # Создаем отдельные колонки для анализа
            df['num_transactions_last_hour'] = df['last_hour_activity'].apply(lambda x: x['num_transactions'] if pd.notna(x) else 0)
            df['total_amount_last_hour'] = df['last_hour_activity'].apply(lambda x: x['total_amount'] if pd.notna(x) else 0)
            df['unique_merchants_last_hour'] = df['last_hour_activity'].apply(lambda x: x['unique_merchants'] if pd.notna(x) else 0)
            df['unique_countries_last_hour'] = df['last_hour_activity'].apply(lambda x: x['unique_countries'] if pd.notna(x) else 0)
            df['max_single_amount_last_hour'] = df['last_hour_activity'].apply(lambda x: x['max_single_amount'] if pd.notna(x) else 0)
            
            results.append("Статистика активности за последний час:")
            activity_cols = ['num_transactions_last_hour', 'total_amount_last_hour', 
                            'unique_merchants_last_hour', 'unique_countries_last_hour', 
                            'max_single_amount_last_hour']
            
            for col in activity_cols:
                results.append(f"\n{col}:")
                results.append(f"  Среднее значение: {df[col].mean():.2f}")
                results.append(f"  Медианное значение: {df[col].median():.2f}")
                results.append(f"  Максимальное значение: {df[col].max():.2f}")
                
                # Анализ корреляции с мошенничеством
                fraud_corr = df[col].corr(df['is_fraud'])
                results.append(f"  Корреляция с мошенничеством: {fraud_corr:.3f}")
            
            # Анализ аномальной активности
            results.append("\nАнализ аномальной активности:")
            high_activity = df[df['num_transactions_last_hour'] > df['num_transactions_last_hour'].quantile(0.95)]
            results.append(f"  Транзакции с высокой активностью (>95%): {len(high_activity):,}")
            results.append(f"  Доля мошенничества среди них: {high_activity['is_fraud'].mean():.3f}")
        else:
            results.append("Данные о активности за последний час недоступны")
        
        return "\n".join(results)
        
    def transaction_amount_distribution_analysis(self):
        """Анализ распределения сумм транзакций"""
        # Проверяем кэш
        if 'transaction_amount_distribution_analysis' in self.analysis_cache:
            return self.analysis_cache['transaction_amount_distribution_analysis']
            
        df = self.data_loader.transactions_df
        results = []
        
        results.append("\n" + "-"*40)
        results.append("РАСПРЕДЕЛЕНИЕ СУММ ТРАНЗАКЦИЙ")
        results.append("-"*40)
        
        # Разделяем данные на легитимные и мошеннические транзакции
        legit_transactions = df[df['is_fraud'] == False]['amount']
        fraud_transactions = df[df['is_fraud'] == True]['amount']
        
        results.append(f"Всего легитимных транзакций: {len(legit_transactions):,}")
        results.append(f"Всего мошеннических транзакций: {len(fraud_transactions):,}")
        
        # Статистика по легитимным транзакциям
        results.append("\nСтатистика легитимных транзакций:")
        results.append(f"  Минимальная сумма: {legit_transactions.min():,.2f} ₽")
        results.append(f"  Максимальная сумма: {legit_transactions.max():,.2f} ₽")
        results.append(f"  Средняя сумма: {legit_transactions.mean():,.2f} ₽")
        results.append(f"  Медианная сумма: {legit_transactions.median():,.2f} ₽")
        results.append(f"  Стандартное отклонение: {legit_transactions.std():,.2f} ₽")
        
        # Квартили для легитимных транзакций
        legit_quartiles = legit_transactions.quantile([0.25, 0.5, 0.75])
        results.append(f"  Q1 (25%): {legit_quartiles[0.25]:,.2f} ₽")
        results.append(f"  Q2 (50%): {legit_quartiles[0.5]:,.2f} ₽")
        results.append(f"  Q3 (75%): {legit_quartiles[0.75]:,.2f} ₽")
        
        # Статистика по мошенническим транзакциям
        results.append("\nСтатистика мошеннических транзакций:")
        results.append(f"  Минимальная сумма: {fraud_transactions.min():,.2f} ₽")
        results.append(f"  Максимальная сумма: {fraud_transactions.max():,.2f} ₽")
        results.append(f"  Средняя сумма: {fraud_transactions.mean():,.2f} ₽")
        results.append(f"  Медианная сумма: {fraud_transactions.median():,.2f} ₽")
        results.append(f"  Стандартное отклонение: {fraud_transactions.std():,.2f} ₽")
        
        # Квартили для мошеннических транзакций
        fraud_quartiles = fraud_transactions.quantile([0.25, 0.5, 0.75])
        results.append(f"  Q1 (25%): {fraud_quartiles[0.25]:,.2f} ₽")
        results.append(f"  Q2 (50%): {fraud_quartiles[0.5]:,.2f} ₽")
        results.append(f"  Q3 (75%): {fraud_quartiles[0.75]:,.2f} ₽")
        
        # Сравнительный анализ
        results.append("\nСравнительный анализ:")
        amount_diff = legit_transactions.mean() - fraud_transactions.mean()
        if amount_diff > 0:
            results.append(f"  Легитимные транзакции в среднем на {amount_diff:,.2f} ₽ больше мошеннических")
        else:
            results.append(f"  Мошеннические транзакции в среднем на {abs(amount_diff):,.2f} ₽ больше легитимных")
        
        # Анализ диапазонов сумм
        results.append("\nАнализ диапазонов сумм:")
        
        # Определяем границы для анализа (используем 95-й процентиль для избежания выбросов)
        legit_95th = legit_transactions.quantile(0.95)
        fraud_95th = fraud_transactions.quantile(0.95)
        
        # Анализируем транзакции в разных диапазонах
        ranges = [
            (0, 1000, "0 - 1,000 ₽"),
            (1000, 5000, "1,000 - 5,000 ₽"),
            (5000, 10000, "5,000 - 10,000 ₽"),
            (10000, 50000, "10,000 - 50,000 ₽"),
            (50000, float('inf'), "Более 50,000 ₽")
        ]
        
        for min_amount, max_amount, range_name in ranges:
            if max_amount == float('inf'):
                legit_count = len(legit_transactions[legit_transactions >= min_amount])
                fraud_count = len(fraud_transactions[fraud_transactions >= min_amount])
            else:
                legit_count = len(legit_transactions[(legit_transactions >= min_amount) & (legit_transactions < max_amount)])
                fraud_count = len(fraud_transactions[(fraud_transactions >= min_amount) & (fraud_transactions < max_amount)])
            
            total_count = legit_count + fraud_count
            if total_count > 0:
                fraud_rate = fraud_count / total_count * 100
                results.append(f"  {range_name}: {total_count:,} транзакций, из них мошеннических: {fraud_count:,} ({fraud_rate:.1f}%)")
        
        # Анализ выбросов
        results.append("\nАнализ выбросов:")
        legit_outliers = legit_transactions[legit_transactions > legit_transactions.quantile(0.99)]
        fraud_outliers = fraud_transactions[fraud_transactions > fraud_transactions.quantile(0.99)]
        
        results.append(f"  Легитимные транзакции > 99%: {len(legit_outliers):,} (макс: {legit_outliers.max():,.2f} ₽)")
        results.append(f"  Мошеннические транзакции > 99%: {len(fraud_outliers):,} (макс: {fraud_outliers.max():,.2f} ₽)")
        
        # Сохраняем структурированные данные для графиков
        amount_data = {
            'text_result': "\n".join(results),
            'legit_transactions': legit_transactions,
            'fraud_transactions': fraud_transactions,
            'legit_quartiles': legit_quartiles,
            'fraud_quartiles': fraud_quartiles
        }
        
        return self._cache_result('transaction_amount_distribution_analysis', amount_data)

class VisualizationEngine:
    """Класс для создания визуализаций"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def create_fraud_distribution_plot(self, analysis_data=None):
        """Создание графиков распределения мошенничества"""
        if analysis_data is None:
            raise ValueError("Необходимы данные анализа для создания графиков")
            
        figures = []
        
        # График 1: Распределение транзакций по типу
        fig1 = Figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        fraud_counts = analysis_data['fraud_counts']
        colors = ['#2E8B57', '#DC143C']
        ax1.pie(fraud_counts.values, labels=['Легитимные', 'Мошеннические'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Распределение транзакций по типу\n(легитимные vs мошеннические)', 
                     fontsize=14, fontweight='bold', pad=20)
        fig1.tight_layout()
        figures.append(fig1)
        
        # График 2: Доля мошеннических транзакций по категориям вендоров
        fig2 = Figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        vendor_fraud = analysis_data['vendor_fraud']['Доля мошенничества']
        ax2.barh(range(len(vendor_fraud)), vendor_fraud.values, color='#FF6B6B')
        ax2.set_yticks(range(len(vendor_fraud)))
        ax2.set_yticklabels(vendor_fraud.index)
        ax2.set_title('Доля мошеннических транзакций по категориям вендоров', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Автоматическое масштабирование оси X
        max_val = vendor_fraud.max()
        if max_val > 0:
            exp = math.floor(math.log10(max_val))
            step = 10 ** exp
            if max_val / step > 5:
                step = step * 2
            elif max_val / step > 2:
                step = step * 1.5
            upper_limit = math.ceil(max_val / step) * step
            ax2.set_xlim(0, upper_limit)
        
        fig2.tight_layout()
        figures.append(fig2)
        
        # График 3: Доля мошеннических транзакций по типам карт
        fig3 = Figure(figsize=(10, 8))
        ax3 = fig3.add_subplot(1, 1, 1)
        card_fraud = analysis_data['card_fraud']['Доля мошенничества']
        ax3.barh(range(len(card_fraud)), card_fraud.values, color='#4ECDC4')
        ax3.set_yticks(range(len(card_fraud)))
        ax3.set_yticklabels([f"{card[:20]}..." if len(card) > 20 else card for card in card_fraud.index])
        ax3.set_title('Доля мошеннических транзакций по типам карт', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Автоматическое масштабирование оси X
        max_val = card_fraud.max()
        if max_val > 0:
            exp = math.floor(math.log10(max_val))
            step = 10 ** exp
            if max_val / step > 5:
                step = step * 2
            elif max_val / step > 2:
                step = step * 1.5
            upper_limit = math.ceil(max_val / step) * step
            ax3.set_xlim(0, upper_limit)
        
        fig3.tight_layout()
        figures.append(fig3)
        
        # График 4: Доля мошеннических транзакций по каналам
        fig4 = Figure(figsize=(10, 8))
        ax4 = fig4.add_subplot(1, 1, 1)
        channel_fraud = analysis_data['channel_fraud']
        ax4.bar(range(len(channel_fraud)), channel_fraud.values, color='#45B7D1')
        ax4.set_xticks(range(len(channel_fraud)))
        ax4.set_xticklabels(channel_fraud.index, rotation=45)
        ax4.set_title('Доля мошеннических транзакций по каналам', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Автоматическое масштабирование оси Y
        max_val = channel_fraud.max()
        if max_val > 0:
            exp = math.floor(math.log10(max_val))
            step = 10 ** exp
            if max_val / step > 5:
                step = step * 2
            elif max_val / step > 2:
                step = step * 1.5
            upper_limit = math.ceil(max_val / step) * step
            ax4.set_ylim(0, upper_limit)
        
        fig4.tight_layout()
        figures.append(fig4)
        
        return figures
        
    def create_temporal_plots(self, analysis_data=None):
        """Создание графиков временных паттернов"""
        if analysis_data is None:
            raise ValueError("Необходимы данные анализа для создания графиков")
            
        figures = []
        
        # График 1: Доля мошеннических транзакций по часам
        fig1 = Figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        hourly_fraud = analysis_data['hourly_fraud']
        ax1.plot(hourly_fraud.index, hourly_fraud.values, marker='o', linewidth=2, markersize=6, color='#FF6B6B')
        ax1.set_xlabel('Час дня (0-23)')
        ax1.set_ylabel('Доля мошеннических транзакций')
        ax1.set_title('Доля мошеннических транзакций по часам дня', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Автоматическое масштабирование оси Y
        max_val = hourly_fraud.max()
        if max_val > 0:
            exp = math.floor(math.log10(max_val))
            step = 10 ** exp
            if max_val / step > 5:
                step = step * 2
            elif max_val / step > 2:
                step = step * 1.5
            upper_limit = math.ceil(max_val / step) * step
            ax1.set_ylim(0, upper_limit)
        
        fig1.tight_layout()
        figures.append(fig1)
        
        # График 2: Доля мошеннических транзакций по дням недели
        fig2 = Figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        daily_fraud = analysis_data['daily_fraud']
        day_names_ru = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        ax2.bar(day_names_ru, daily_fraud.values, color='#4ECDC4')
        ax2.set_ylabel('Доля мошеннических транзакций')
        ax2.set_title('Доля мошеннических транзакций по дням недели', fontsize=14, fontweight='bold', pad=20)
        
        # Автоматическое масштабирование оси Y
        max_val = daily_fraud.max()
        if max_val > 0:
            exp = math.floor(math.log10(max_val))
            step = 10 ** exp
            if max_val / step > 5:
                step = step * 2
            elif max_val / step > 2:
                step = step * 1.5
            upper_limit = math.ceil(max_val / step) * step
            ax2.set_ylim(0, upper_limit)
        
        fig2.tight_layout()
        figures.append(fig2)
        
        # График 3: Общее количество транзакций по часам
        fig3 = Figure(figsize=(10, 8))
        ax3 = fig3.add_subplot(1, 1, 1)
        hourly_volume = analysis_data['hourly_volume']
        ax3.bar(hourly_volume.index, hourly_volume.values, color='#96CEB4', alpha=0.7)
        ax3.set_xlabel('Час дня (0-23)')
        ax3.set_ylabel('Количество транзакций')
        ax3.set_title('Общее количество транзакций по часам дня', fontsize=14, fontweight='bold', pad=20)
        
        # Автоматическое масштабирование оси Y
        max_val = hourly_volume.max()
        if max_val > 0:
            exp = math.floor(math.log10(max_val))
            step = 10 ** exp
            if max_val / step > 5:
                step = step * 2
            elif max_val / step > 2:
                step = step * 1.5
            upper_limit = math.ceil(max_val / step) * step
            ax3.set_ylim(0, upper_limit)
        
        fig3.tight_layout()
        figures.append(fig3)
        
        return figures

class ModelTrainingThread(QThread):
    """Поток для обучения модели машинного обучения"""
    
    # Сигналы для обновления GUI
    progress_signal = pyqtSignal(str)  # Сообщения о прогрессе
    finished_signal = pyqtSignal(dict)  # Результат обучения
    error_signal = pyqtSignal(str)  # Ошибка
    
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.mutex = QMutex()
        self._is_running = False
        
    def run(self):
        """Основной метод выполнения обучения в отдельном потоке"""
        self._is_running = True
        try:
            self.mutex.lock()
            
            # Проверяем, не был ли поток остановлен
            if not self._is_running:
                return
            
            # Импортируем необходимые библиотеки
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.utils.class_weight import compute_class_weight
            
            self.progress_signal.emit("📊 Подготавливаю данные для моделирования...")
            
            # Проверяем, не был ли поток остановлен
            if not self._is_running:
                return
            
            # Подготавливаем данные
            df = self.data_loader.transactions_df.copy()
            
            # Конвертируем timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6])
            
            # Извлекаем данные из last_hour_activity
            if 'last_hour_activity' in df.columns:
                try:
                    df['num_transactions_last_hour'] = df['last_hour_activity'].apply(
                        lambda x: x.get('num_transactions', 0) if pd.notna(x) and isinstance(x, dict) else 0)
                    df['total_amount_last_hour'] = df['last_hour_activity'].apply(
                        lambda x: x.get('total_amount', 0) if pd.notna(x) and isinstance(x, dict) else 0)
                    df['unique_merchants_last_hour'] = df['last_hour_activity'].apply(
                        lambda x: x.get('unique_merchants', 0) if pd.notna(x) and isinstance(x, dict) else 0)
                    df['unique_countries_last_hour'] = df['last_hour_activity'].apply(
                        lambda x: x.get('unique_countries', 0) if pd.notna(x) and isinstance(x, dict) else 0)
                    df['max_single_amount_last_hour'] = df['last_hour_activity'].apply(
                        lambda x: x.get('max_single_amount', 0) if pd.notna(x) and isinstance(x, dict) else 0)
                except Exception as e:
                    self.progress_signal.emit(f"⚠️  Предупреждение: Ошибка при обработке last_hour_activity: {str(e)}")
                    # Устанавливаем значения по умолчанию
                    df['num_transactions_last_hour'] = 0
                    df['total_amount_last_hour'] = 0
                    df['unique_merchants_last_hour'] = 0
                    df['unique_countries_last_hour'] = 0
                    df['max_single_amount_last_hour'] = 0
            else:
                # Если колонки нет, создаем с нулевыми значениями
                df['num_transactions_last_hour'] = 0
                df['total_amount_last_hour'] = 0
                df['unique_merchants_last_hour'] = 0
                df['unique_countries_last_hour'] = 0
                df['max_single_amount_last_hour'] = 0
            
            # Инжиниринг признаков
            self.progress_signal.emit("🔧 Выполняю инжиниринг признаков...")
            
            # Кодируем категориальные переменные
            categorical_columns = ['vendor_category', 'vendor_type', 'currency', 'country', 
                                  'city_size', 'card_type', 'device', 'channel']
            
            for col in categorical_columns:
                if col in df.columns:
                    try:
                        le = LabelEncoder()
                        # Заполняем NaN значения перед кодированием
                        df[col] = df[col].fillna('unknown')
                        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    except Exception as e:
                        self.progress_signal.emit(f"⚠️  Предупреждение: Ошибка при кодировании {col}: {str(e)}")
                        # Создаем простую нумерацию в случае ошибки
                        df[f'{col}_encoded'] = pd.Categorical(df[col].fillna('unknown')).codes
            
            # Создаем новые признаки
            try:
                df['amount_log'] = np.log1p(df['amount'].fillna(0))
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                
                # Безопасное вычисление квантилей
                amount_95 = df['amount'].quantile(0.95) if len(df) > 0 else 0
                amount_05 = df['amount'].quantile(0.05) if len(df) > 0 else 0
                
                df['is_high_amount'] = (df['amount'] > amount_95).astype(int)
                df['is_low_amount'] = (df['amount'] < amount_05).astype(int)
                df['activity_intensity'] = df['num_transactions_last_hour'] * df['total_amount_last_hour']
                df['geographic_spread'] = df['unique_countries_last_hour'] * df['unique_merchants_last_hour']
            except Exception as e:
                self.progress_signal.emit(f"⚠️  Предупреждение: Ошибка при создании признаков: {str(e)}")
                # Устанавливаем значения по умолчанию
                df['amount_log'] = 0
                df['hour_sin'] = 0
                df['hour_cos'] = 0
                df['is_high_amount'] = 0
                df['is_low_amount'] = 0
                df['activity_intensity'] = 0
                df['geographic_spread'] = 0
            
            # Выбираем числовые признаки
            numeric_features = [
                'amount', 'amount_log', 'hour', 'hour_sin', 'hour_cos', 'is_weekend',
                'is_outside_home_country', 'is_high_risk_vendor', 'is_card_present',
                'num_transactions_last_hour', 'total_amount_last_hour', 'unique_merchants_last_hour',
                'unique_countries_last_hour', 'max_single_amount_last_hour',
                'is_high_amount', 'is_low_amount', 'activity_intensity', 'geographic_spread'
            ]
            
            # Добавляем закодированные категориальные признаки
            encoded_features = [col for col in df.columns if col.endswith('_encoded')]
            numeric_features.extend(encoded_features)
            numeric_features = [col for col in numeric_features if col in df.columns]
            
            # Подготавливаем данные для моделирования
            df_clean = df[numeric_features + ['is_fraud']].dropna()
            X = df_clean[numeric_features]
            y = df_clean['is_fraud']
            
            # Проверяем, что у нас есть данные для обучения
            if len(X) == 0:
                raise ValueError("Нет данных для обучения модели после очистки")
            
            if len(y.unique()) < 2:
                raise ValueError("Недостаточно классов для обучения модели (нужно минимум 2 класса)")
            
            self.progress_signal.emit(f"✅ Подготовлено {len(X):,} транзакций с {len(numeric_features)} признаками")
            self.progress_signal.emit(f"📈 Распределение классов: {y.value_counts().to_dict()}")
            
            # Разделяем данные
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Масштабируем признаки
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Проверяем, не был ли поток остановлен
            if not self._is_running:
                return
            
            # Обучаем модели с оптимизацией
            self.progress_signal.emit("\n🤖 Обучаю модели с использованием всех доступных ядер процессора...")
            
            # Определяем количество ядер для параллельной обработки
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), 8)  # Ограничиваем максимум 8 ядрами
            
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    class_weight='balanced',
                    n_jobs=n_jobs,  # Используем все доступные ядра
                    verbose=0
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100, 
                    random_state=42
                    # GradientBoostingClassifier не поддерживает n_jobs
                ),
                'Logistic Regression': LogisticRegression(
                    random_state=42, 
                    class_weight='balanced', 
                    max_iter=1000,
                    n_jobs=n_jobs  # Используем все доступные ядра
                )
            }
            
            results = {}
            
            for name, model in models.items():
                try:
                    self.progress_signal.emit(f"\n📊 Обучаю {name} на {n_jobs} ядрах...")
                    
                    if name == 'Logistic Regression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Оцениваем модель
                    accuracy = (y_pred == y_test).mean()
                    auc = roc_auc_score(y_test, y_pred_proba)
                    
                    # Матрица ошибок
                    cm = confusion_matrix(y_test, y_pred)
                    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
                    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    self.progress_signal.emit(f"  ✅ Точность: {accuracy:.4f}")
                    self.progress_signal.emit(f"  📊 AUC-ROC: {auc:.4f}")
                    self.progress_signal.emit(f"  🎯 Precision: {precision:.4f}")
                    self.progress_signal.emit(f"  🔍 Recall: {recall:.4f}")
                    self.progress_signal.emit(f"  ⚖️  F1-Score: {f1:.4f}")
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'auc': auc,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'model': model,
                        'scaler': scaler if name == 'Logistic Regression' else None,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                except Exception as e:
                    self.progress_signal.emit(f"❌ Ошибка при обучении {name}: {str(e)}")
                    continue
            
            # Проверяем, что хотя бы одна модель была обучена
            if not results:
                raise ValueError("Не удалось обучить ни одной модели")
            
            # Находим лучшую модель
            best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
            best_result = results[best_model_name]
            
            self.progress_signal.emit(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
            self.progress_signal.emit(f"   AUC-ROC: {best_result['auc']:.4f}")
            self.progress_signal.emit(f"   Точность: {best_result['accuracy']:.4f}")
            self.progress_signal.emit(f"   F1-Score: {best_result['f1']:.4f}")
            
            # Сохраняем лучшую модель
            import joblib
            model_filename = f'best_fraud_detection_model_{best_model_name.lower().replace(" ", "_")}.joblib'
            joblib.dump(best_result['model'], model_filename)
            if best_result['scaler']:
                scaler_filename = f'scaler_{best_model_name.lower().replace(" ", "_")}.joblib'
                joblib.dump(best_result['scaler'], scaler_filename)
            
            self.progress_signal.emit(f"\n💾 Модель сохранена в файл: {model_filename}")
            
            # Бизнес-инсайты
            cm = confusion_matrix(y_test, best_result['y_pred'])
            total_fraud = cm[1,0] + cm[1,1]
            total_legitimate = cm[0,0] + cm[0,1]
            fraud_detection_rate = cm[1,1] / total_fraud if total_fraud > 0 else 0
            false_alarm_rate = cm[0,1] / total_legitimate if total_legitimate > 0 else 0
            
            self.progress_signal.emit(f"\n📈 БИЗНЕС-МЕТРИКИ:")
            self.progress_signal.emit(f"   Доля обнаруженного мошенничества: {fraud_detection_rate:.1%}")
            self.progress_signal.emit(f"   Доля ложных срабатываний: {false_alarm_rate:.1%}")
            
            # Рекомендации
            self.progress_signal.emit(f"\n💡 РЕКОМЕНДАЦИИ:")
            if fraud_detection_rate < 0.8:
                self.progress_signal.emit("   ⚠️  Необходимо улучшить детекцию мошенничества")
            else:
                self.progress_signal.emit("   ✅ Хороший уровень детекции мошенничества")
            
            if false_alarm_rate > 0.1:
                self.progress_signal.emit("   ⚠️  Высокий уровень ложных срабатываний - требуется оптимизация")
            else:
                self.progress_signal.emit("   ✅ Приемлемый уровень ложных срабатываний")
            
            if best_result['auc'] > 0.9:
                self.progress_signal.emit("   🏆 Отличное качество модели")
            elif best_result['auc'] > 0.8:
                self.progress_signal.emit("   ✅ Хорошее качество модели")
            else:
                self.progress_signal.emit("   ⚠️  Требуется улучшение модели")
            
            self.progress_signal.emit(f"\n✅ Обучение модели завершено успешно!")
            
            # Проверяем, не был ли поток остановлен перед отправкой результата
            if not self._is_running:
                return
            
            # Отправляем результат
            self.finished_signal.emit({
                'success': True,
                'best_model_name': best_model_name,
                'best_result': best_result,
                'model_filename': model_filename,
                'scaler_filename': scaler_filename if best_result['scaler'] else None
            })
            
        except Exception as e:
            if self._is_running:  # Отправляем ошибку только если поток не был остановлен
                self.error_signal.emit(f"❌ Ошибка при обучении модели: {str(e)}\nПроверьте, что данные загружены корректно и содержат необходимые поля.")
        finally:
            self._is_running = False
            self.mutex.unlock()
    
    def stop(self):
        """Остановка потока"""
        self._is_running = False


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.data_loader = DataLoader()
        self.analysis_engine = AnalysisEngine(self.data_loader)
        self.visualization_engine = VisualizationEngine(self.data_loader)
        
        # Поток для обучения модели
        self.training_thread = None
        
        # Отдельные списки для хранения графиков разных типов
        self.distribution_plots = []
        self.temporal_plots = []
        self.current_distribution_index = 0
        self.current_temporal_index = 0
        
        self.init_ui()
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Анализ мошеннических транзакций")
        self.setGeometry(100, 100, 1400, 900)
        
        # Устанавливаем стиль для всего приложения
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                background-color: #f8f9fa;
            }
        """)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Заголовок
        title_label = QLabel("🔍 АНАЛИЗ МОШЕННИЧЕСКИХ ТРАНЗАКЦИЙ")
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 3px; padding: 5px; background-color: #e3f2fd; border: 2px solid #2196f3; border-radius: 6px;")
        main_layout.addWidget(title_label)
        
        # Разделитель
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #bdc3c7;
                border-radius: 2px;
            }
        """)
        main_layout.addWidget(splitter)
        
        # Левая панель
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        # Группа загрузки файлов
        upload_group = QGroupBox("📁 ЗАГРУЗКА ДАННЫХ")
        upload_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
                background-color: #f8f9fa;
            }
        """)
        
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setSpacing(15)
        upload_layout.setContentsMargins(20, 20, 20, 20)
        
        # Кнопки загрузки (по центру)
        self.transactions_btn = QPushButton("📊 Загрузить транзакции")
        self.transactions_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #1f5f8b);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1f5f8b, stop:1 #154360);
            }
        """)
        
        # Статус загрузки транзакций (под кнопкой)
        self.transactions_status = QLabel("❌ Файл транзакций не загружен")
        self.transactions_status.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px; background-color: #fdf2f2; border-radius: 5px; border: 1px solid #f5c6cb; text-align: center;")
        self.transactions_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.currency_btn = QPushButton("💱 Загрузить валюты")
        self.currency_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c0392b, stop:1 #a93226);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a93226, stop:1 #7b241c);
            }
        """)
        
        # Статус загрузки валют (под кнопкой)
        self.currency_status = QLabel("❌ Файл валют не загружен")
        self.currency_status.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px; background-color: #fdf2f2; border-radius: 5px; border: 1px solid #f5c6cb; text-align: center;")
        self.currency_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Размещаем элементы вертикально: кнопка -> статус -> кнопка -> статус
        upload_layout.addWidget(self.transactions_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(self.transactions_status, alignment=Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(self.currency_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(self.currency_status, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Подключаем сигналы
        self.transactions_btn.clicked.connect(self.load_transactions_file)
        self.currency_btn.clicked.connect(self.load_currency_file)
        
        left_layout.addWidget(upload_group)
        
        # Группа анализа
        analysis_group = QGroupBox("🔍 АНАЛИЗ ДАННЫХ")
        analysis_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
                background-color: #f8f9fa;
            }
        """)
        
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(10)
        analysis_layout.setContentsMargins(20, 20, 20, 20)
        
        analysis_btn = QPushButton("📈 Запустить анализ")
        analysis_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27ae60, stop:1 #229954);
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #229954, stop:1 #1e8449);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e8449, stop:1 #196f3d);
            }
        """)
        analysis_btn.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(analysis_btn)
        
        # Кнопка сохранения результатов анализа
        save_analysis_btn = QPushButton("💾 Сохранить анализ")
        save_analysis_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8e44ad, stop:1 #7d3c98);
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7d3c98, stop:1 #6c3483);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6c3483, stop:1 #5b2c6b);
            }
        """)
        save_analysis_btn.clicked.connect(self.save_analysis_results)
        analysis_layout.addWidget(save_analysis_btn)
        
        left_layout.addWidget(analysis_group)
        
        # Группа визуализации
        viz_group = QGroupBox("📊 ВИЗУАЛИЗАЦИЯ")
        viz_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
                background-color: #f8f9fa;
            }
        """)
        
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setSpacing(10)
        viz_layout.setContentsMargins(20, 20, 20, 20)
        
        viz_btn = QPushButton("📊 Создать графики")
        viz_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f39c12, stop:1 #e67e22);
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e67e22, stop:1 #d35400);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d35400, stop:1 #ba4a00);
            }
        """)
        viz_btn.clicked.connect(self.create_distribution_plot)
        viz_layout.addWidget(viz_btn)
        
        temporal_btn = QPushButton("⏰ Временной анализ")
        temporal_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e67e22, stop:1 #d35400);
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d35400, stop:1 #ba4a00);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ba4a00, stop:1 #a04000);
            }
        """)
        temporal_btn.clicked.connect(self.create_temporal_plots)
        viz_layout.addWidget(temporal_btn)
        
        # Кнопка очистки графиков
        clear_btn = QPushButton("🗑️ Очистить графики")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c0392b, stop:1 #a93226);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a93226, stop:1 #7b241c);
            }
        """)
        clear_btn.clicked.connect(self.clear_plots)
        viz_layout.addWidget(clear_btn)
        
        left_layout.addWidget(viz_group)
        
        # Группа модели
        model_group = QGroupBox("🤖 МАШИННОЕ ОБУЧЕНИЕ")
        model_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                color: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
                background-color: #f8f9fa;
            }
        """)
        
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(20, 20, 20, 20)
        
        model_btn = QPushButton("🎯 Обучить модель")
        model_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9b59b6, stop:1 #8e44ad);
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8e44ad, stop:1 #7d3c98);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7d3c98, stop:1 #6c3483);
            }
        """)
        model_btn.clicked.connect(self.train_model)
        model_layout.addWidget(model_btn)
        
        # Кнопка предсказания с моделью
        predict_btn = QPushButton("🔮 Предсказать")
        predict_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4B0082, stop:1 #663399);
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #663399, stop:1 #4B0082);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4B0082, stop:1 #2E0854);
            }
        """)
        predict_btn.clicked.connect(self.predict_with_model)
        model_layout.addWidget(predict_btn)
        
        left_layout.addWidget(model_group)
        
        # Добавляем растягивающийся элемент для центрирования
        left_layout.addStretch()
        
        # Правая панель (увеличиваем размер)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        # Создаем вкладки для результатов
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background-color: #ffffff;
                padding: 10px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ecf0f1, stop:1 #bdc3c7);
                color: #2c3e50;
                padding: 8px 16px;
                margin: 2px;
                border: 1px solid #95a5a6;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                min-width: 80px;
            }
            QTabBar::tab:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #d5dbdb, stop:1 #aeb6b7);
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: 1px solid #2980b9;
            }
        """)
        
        # Вкладка для текстовых результатов
        self.text_tab = QWidget()
        text_layout = QVBoxLayout(self.text_tab)
        text_layout.setContentsMargins(5, 5, 5, 5)
        
        self.results_text = QTextEdit()
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #2c3e50;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                padding: 10px;
                selection-background-color: #3498db;
                selection-color: white;
            }
        """)
        self.results_text.setMinimumWidth(600)  # Увеличиваем минимальную ширину
        text_layout.addWidget(self.results_text)
        
        # Вкладка для графиков распределения
        self.distribution_tab = QWidget()
        distribution_layout = QVBoxLayout(self.distribution_tab)
        distribution_layout.setContentsMargins(5, 5, 5, 5)
        
        # Панель навигации для графиков распределения
        self.distribution_navigation = QWidget()
        dist_nav_layout = QHBoxLayout(self.distribution_navigation)
        dist_nav_layout.setContentsMargins(0, 10, 0, 10)
        
        # Кнопки навигации для графиков распределения
        self.dist_prev_btn = QPushButton("◀ Предыдущий")
        self.dist_prev_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #95a5a6, stop:1 #7f8c8d);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7f8c8d, stop:1 #6c7b7d);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #95a5a6;
            }
        """)
        self.dist_prev_btn.clicked.connect(self.show_previous_distribution_plot)
        self.dist_prev_btn.setEnabled(False)
        
        # Информация о текущем графике распределения
        self.dist_plot_info = QLabel("График 0 из 0")
        self.dist_plot_info.setStyleSheet("""
            color: #2c3e50;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
        """)
        self.dist_plot_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.dist_next_btn = QPushButton("Следующий ▶")
        self.dist_next_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #95a5a6, stop:1 #7f8c8d);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7f8c8d, stop:1 #6c7b7d);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #95a5a6;
            }
        """)
        self.dist_next_btn.clicked.connect(self.show_next_distribution_plot)
        self.dist_next_btn.setEnabled(False)
        
        dist_nav_layout.addWidget(self.dist_prev_btn)
        dist_nav_layout.addWidget(self.dist_plot_info)
        dist_nav_layout.addWidget(self.dist_next_btn)
        
        # Скрываем панель навигации изначально
        self.distribution_navigation.hide()
        
        # Создаем виджет для отображения графиков распределения
        self.distribution_plot_widget = QWidget()
        self.distribution_plot_layout = QVBoxLayout(self.distribution_plot_widget)
        self.distribution_plot_layout.setContentsMargins(5, 5, 5, 5)
        
        distribution_layout.addWidget(self.distribution_navigation)
        distribution_layout.addWidget(self.distribution_plot_widget)
        
        # Вкладка для временных графиков
        self.temporal_tab = QWidget()
        temporal_layout = QVBoxLayout(self.temporal_tab)
        temporal_layout.setContentsMargins(5, 5, 5, 5)
        
        # Панель навигации для временных графиков
        self.temporal_navigation = QWidget()
        temp_nav_layout = QHBoxLayout(self.temporal_navigation)
        temp_nav_layout.setContentsMargins(0, 10, 0, 10)
        
        # Кнопки навигации для временных графиков
        self.temp_prev_btn = QPushButton("◀ Предыдущий")
        self.temp_prev_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #95a5a6, stop:1 #7f8c8d);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7f8c8d, stop:1 #6c7b7d);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #95a5a6;
            }
        """)
        self.temp_prev_btn.clicked.connect(self.show_previous_temporal_plot)
        self.temp_prev_btn.setEnabled(False)
        
        # Информация о текущем временном графике
        self.temp_plot_info = QLabel("График 0 из 0")
        self.temp_plot_info.setStyleSheet("""
            color: #2c3e50;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
        """)
        self.temp_plot_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.temp_next_btn = QPushButton("Следующий ▶")
        self.temp_next_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #95a5a6, stop:1 #7f8c8d);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7f8c8d, stop:1 #6c7b7d);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #95a5a6;
            }
        """)
        self.temp_next_btn.clicked.connect(self.show_next_temporal_plot)
        self.temp_next_btn.setEnabled(False)
        
        temp_nav_layout.addWidget(self.temp_prev_btn)
        temp_nav_layout.addWidget(self.temp_plot_info)
        temp_nav_layout.addWidget(self.temp_next_btn)
        
        # Скрываем панель навигации изначально
        self.temporal_navigation.hide()
        
        # Создаем виджет для отображения временных графиков
        self.temporal_plot_widget = QWidget()
        self.temporal_plot_layout = QVBoxLayout(self.temporal_plot_widget)
        self.temporal_plot_layout.setContentsMargins(5, 5, 5, 5)
        
        temporal_layout.addWidget(self.temporal_navigation)
        temporal_layout.addWidget(self.temporal_plot_widget)
        
        # Добавляем вкладки
        self.results_tabs.addTab(self.text_tab, "📝 Текст")
        self.results_tabs.addTab(self.distribution_tab, "📊 Графики распределения")
        self.results_tabs.addTab(self.temporal_tab, "⏰ Временные графики")
        
        right_layout.addWidget(self.results_tabs)
        
        # Добавляем панели в разделитель с пропорциями 1:3 (левая:правая)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])  # Устанавливаем начальные размеры
        
        # Устанавливаем растягивающиеся свойства
        splitter.setStretchFactor(0, 0)  # Левая панель не растягивается
        splitter.setStretchFactor(1, 1)  # Правая панель растягивается
        
    def load_transactions_file(self):
        """Загрузка файла транзакций"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными транзакций", "", 
            "Parquet files (*.parquet);;All files (*)"
        )
        
        if file_path:
            if self.data_loader.load_transactions(file_path):
                self.transactions_status.setText("✅ Файл транзакций загружен")
                self.transactions_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px; background-color: #e8f5e8; border-radius: 5px; border: 1px solid #a8e6cf;")
            else:
                self.transactions_status.setText("❌ Ошибка загрузки файла")
                self.transactions_status.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px; background-color: #fdf2f2; border-radius: 5px; border: 1px solid #f5c6cb;")
                
    def load_currency_file(self):
        """Загрузка файла валют"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными валют", "", 
            "Parquet files (*.parquet);;All files (*)"
        )
        
        if file_path:
            if self.data_loader.load_currency(file_path):
                self.currency_status.setText("✅ Файл валют загружен")
                self.currency_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px; background-color: #e8f5e8; border-radius: 5px; border: 1px solid #a8e6cf;")
            else:
                self.currency_status.setText("❌ Ошибка загрузки файла")
                self.currency_status.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px; background-color: #fdf2f2; border-radius: 5px; border: 1px solid #f5c6cb;")
                
    def run_analysis(self):
        """Запуск анализа данных"""
        if not self.data_loader.check_data_ready():
            self.results_text.setText("❌ Сначала загрузите оба файла данных!")
            return
            
        self.results_text.clear()
        self.results_text.append("🔍 Запускаю анализ данных...\n")
        
        # Базовый обзор
        basic_overview = self.analysis_engine.basic_data_overview()
        self.results_text.append(basic_overview)
        self.results_text.append("\n")
        
        # Анализ мошенничества
        fraud_data = self.analysis_engine.fraud_analysis()
        self.results_text.append(fraud_data['text_result'])
        self.results_text.append("\n")
        
        # Временной анализ
        temporal_data = self.analysis_engine.temporal_analysis()
        self.results_text.append(temporal_data['text_result'])
        self.results_text.append("\n")
        
        # Анализ распределения сумм транзакций (добавляем к временному анализу)
        amount_dist_data = self.analysis_engine.transaction_amount_distribution_analysis()
        self.results_text.append(amount_dist_data['text_result'])
        self.results_text.append("\n")
        
        # Географический анализ
        geo_data = self.analysis_engine.geographical_analysis()
        self.results_text.append(geo_data)
        self.results_text.append("\n")
        
        # Анализ сумм транзакций
        amount_data = self.analysis_engine.amount_analysis()
        self.results_text.append(amount_data)
        self.results_text.append("\n")
        
        # Анализ устройств и каналов
        device_data = self.analysis_engine.device_channel_analysis()
        self.results_text.append(device_data)
        self.results_text.append("\n")
        
        # Анализ активности за последний час
        activity_data = self.analysis_engine.last_hour_activity_analysis()
        self.results_text.append(activity_data)
        
        self.results_text.append("\n✅ Анализ завершен!")
        
    def create_distribution_plot(self):
        """Создание графиков распределения мошенничества"""
        if not self.data_loader.check_data_ready():
            self.results_text.setText("❌ Сначала загрузите оба файла данных!")
            return
            
        try:
            # Очищаем предыдущие графики
            self.clear_plots()
            
            # Получаем данные для анализа
            fraud_data = self.analysis_engine.fraud_analysis()
            
            # Создаем графики
            figures = self.visualization_engine.create_fraud_distribution_plot(fraud_data)
            
            # Добавляем графики в список
            for i, fig in enumerate(figures):
                self.distribution_plots.append({
                    'figure': fig,
                    'title': f'Распределение мошенничества {i+1}',
                    'type': 'distribution'
                })
            
            # Отображаем текущий график
            self.current_distribution_index = 0
            self.display_current_distribution_plot()
            
            # Переключаемся на вкладку с графиками распределения
            self.results_tabs.setCurrentIndex(1)
            
            # Добавляем сообщение в текстовую вкладку
            self.results_text.append("✅ Графики распределения созданы! Переключитесь на вкладку 'Графики распределения' для просмотра.")
            
        except Exception as e:
            error_msg = f"❌ Ошибка при создании графиков: {str(e)}"
            self.results_text.append(error_msg)
            print(f"Error in create_distribution_plot: {e}")
        
    def create_temporal_plots(self):
        """Создание графиков временных паттернов"""
        if not self.data_loader.check_data_ready():
            self.results_text.setText("❌ Сначала загрузите оба файла данных!")
            return
            
        try:
            # Очищаем предыдущие графики
            self.clear_plots()
            
            # Получаем данные для анализа
            analysis_data = self.analysis_engine.temporal_analysis()
            
            # Создаем графики через VisualizationEngine
            figures = self.visualization_engine.create_temporal_plots(analysis_data)
            
            # Добавляем графики в список
            for i, fig in enumerate(figures):
                self.temporal_plots.append({
                    'figure': fig,
                    'title': f'Временной график {i+1}',
                    'type': 'temporal'
                })
            
            # Отображаем текущий график
            self.current_temporal_index = 0
            self.display_current_temporal_plot()
            
            # Переключаемся на вкладку с временными графиками
            self.results_tabs.setCurrentIndex(2)
            
            # Добавляем сообщение в текстовую вкладку
            self.results_text.append("✅ Временные графики созданы! Переключитесь на вкладку 'Временные графики' для просмотра.")
            
        except Exception as e:
            error_msg = f"❌ Ошибка при создании временных графиков: {str(e)}"
            self.results_text.append(error_msg)
            print(f"Error in create_temporal_plots: {e}")
        
    def clear_plots(self):
        """Очистка всех графиков"""
        print(f"Очистка графиков: было {len(self.distribution_plots)} графиков распределения и {len(self.temporal_plots)} временных графиков")
        
        # Закрываем все matplotlib фигуры
        for plot_info in self.distribution_plots:
            if 'figure' in plot_info and plot_info['figure']:
                plt.close(plot_info['figure'])
        for plot_info in self.temporal_plots:
            if 'figure' in plot_info and plot_info['figure']:
                plt.close(plot_info['figure'])
        
        # Очищаем списки графиков
        self.current_distribution_index = 0
        self.distribution_plots.clear()
        self.current_temporal_index = 0
        self.temporal_plots.clear()
        
        # Очищаем layout'ы графиков
        self.clear_distribution_plot_layout()
        self.clear_temporal_plot_layout()
        
        # Скрываем навигацию
        self.distribution_navigation.hide()
        self.temporal_navigation.hide()
        
        # Принудительно обновляем отображение
        self.distribution_plot_widget.update()
        self.temporal_plot_widget.update()
        
        print("Графики очищены")
        
    def clear_distribution_plot_layout(self):
        """Очистка layout графиков распределения"""
        # Удаляем все виджеты из layout графиков распределения
        widgets_removed = 0
        while self.distribution_plot_layout.count() > 0:
            child = self.distribution_plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                widgets_removed += 1
        
        # Принудительно очищаем layout
        self.distribution_plot_layout.update()
        
        # Отладочная информация
        print(f"Очищен layout графиков распределения, удалено {widgets_removed} виджетов")
        
    def clear_temporal_plot_layout(self):
        """Очистка layout временных графиков"""
        # Удаляем все виджеты из layout временных графиков
        widgets_removed = 0
        while self.temporal_plot_layout.count() > 0:
            child = self.temporal_plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                widgets_removed += 1
        
        # Принудительно очищаем layout
        self.temporal_plot_layout.update()
        
        # Отладочная информация
        print(f"Очищен layout временных графиков, удалено {widgets_removed} виджетов")
        
    def display_current_distribution_plot(self):
        """Отображение текущего графика распределения"""
        if not self.distribution_plots:
            self.distribution_navigation.hide()
            return
            
        # Очищаем layout графиков распределения
        self.clear_distribution_plot_layout()
        
        # Показываем навигацию
        self.distribution_navigation.show()
        
        # Получаем текущий график
        current_plot = self.distribution_plots[self.current_distribution_index]
        fig = current_plot['figure']
        title = current_plot['title']
        
        # Создаем новый canvas для matplotlib
        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(800, 600)  # Увеличиваем минимальный размер
        
        # Добавляем canvas в layout графиков
        self.distribution_plot_layout.addWidget(canvas)
        
        # Обновляем информацию о графике
        self.update_distribution_plot_navigation()
        
        # Обновляем заголовок вкладки
        self.distribution_tab.setToolTip(f"График: {title}")
        
        # Принудительно обновляем отображение
        canvas.draw()
        self.distribution_plot_layout.update()
        
        # Принудительно обновляем весь виджет графиков
        self.distribution_plot_widget.update()
        
        # Отладочная информация
        print(f"Отображен график распределения {self.current_distribution_index + 1} из {len(self.distribution_plots)}: {title}")
        print(f"Canvas размер: {canvas.size()}")
        print(f"Layout графиков распределения содержит {self.distribution_plot_layout.count()} виджетов")
        
    def display_current_temporal_plot(self):
        """Отображение текущего временного графика"""
        if not self.temporal_plots:
            self.temporal_navigation.hide()
            return
            
        # Очищаем layout временных графиков
        self.clear_temporal_plot_layout()
        
        # Показываем навигацию
        self.temporal_navigation.show()
        
        # Получаем текущий график
        current_plot = self.temporal_plots[self.current_temporal_index]
        fig = current_plot['figure']
        title = current_plot['title']
        
        # Создаем новый canvas для matplotlib
        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(800, 600)  # Увеличиваем минимальный размер
        
        # Добавляем canvas в layout графиков
        self.temporal_plot_layout.addWidget(canvas)
        
        # Обновляем информацию о графике
        self.update_temporal_plot_navigation()
        
        # Обновляем заголовок вкладки
        self.temporal_tab.setToolTip(f"График: {title}")
        
        # Принудительно обновляем отображение
        canvas.draw()
        self.temporal_plot_layout.update()
        
        # Принудительно обновляем весь виджет графиков
        self.temporal_plot_widget.update()
        
        # Отладочная информация
        print(f"Отображен временной график {self.current_temporal_index + 1} из {len(self.temporal_plots)}: {title}")
        print(f"Canvas размер: {canvas.size()}")
        print(f"Layout временных графиков содержит {self.temporal_plot_layout.count()} виджетов")
        
    def update_distribution_plot_navigation(self):
        """Обновление состояния навигации для графиков распределения"""
        if not self.distribution_plots:
            self.dist_plot_info.setText("График 0 из 0")
            self.dist_prev_btn.setEnabled(False)
            self.dist_next_btn.setEnabled(False)
            return
            
        total_plots = len(self.distribution_plots)
        current_plot = self.current_distribution_index + 1
        
        self.dist_plot_info.setText(f"График {current_plot} из {total_plots}")
        
        # Включаем/выключаем кнопки
        self.dist_prev_btn.setEnabled(self.current_distribution_index > 0)
        self.dist_next_btn.setEnabled(self.current_distribution_index < total_plots - 1)
        
    def update_temporal_plot_navigation(self):
        """Обновление состояния навигации для временных графиков"""
        if not self.temporal_plots:
            self.temp_plot_info.setText("График 0 из 0")
            self.temp_prev_btn.setEnabled(False)
            self.temp_next_btn.setEnabled(False)
            return
            
        total_plots = len(self.temporal_plots)
        current_plot = self.current_temporal_index + 1
        
        self.temp_plot_info.setText(f"График {current_plot} из {total_plots}")
        
        # Включаем/выключаем кнопки
        self.temp_prev_btn.setEnabled(self.current_temporal_index > 0)
        self.temp_next_btn.setEnabled(self.current_temporal_index < total_plots - 1)
        
    def show_previous_distribution_plot(self):
        """Показать предыдущий график распределения"""
        if self.current_distribution_index > 0:
            self.current_distribution_index -= 1
            self.display_current_distribution_plot()
        
    def show_next_distribution_plot(self):
        """Показать следующий график распределения"""
        if self.current_distribution_index < len(self.distribution_plots) - 1:
            self.current_distribution_index += 1
            self.display_current_distribution_plot()
        
    def show_previous_temporal_plot(self):
        """Показать предыдущий временной график"""
        if self.current_temporal_index > 0:
            self.current_temporal_index -= 1
            self.display_current_temporal_plot()
        
    def show_next_temporal_plot(self):
        """Показать следующий временной график"""
        if self.current_temporal_index < len(self.temporal_plots) - 1:
            self.current_temporal_index += 1
            self.display_current_temporal_plot()
        
    def show_plot_window(self, fig, title):
        """Отображение графика в отдельном окне"""
        plot_window = QWidget()
        plot_window.setWindowTitle(title)
        plot_window.setGeometry(200, 200, 900, 700)
        plot_window.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout(plot_window)
        layout.setContentsMargins(15, 15, 15, 15)
        
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        plot_window.show()
    
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        # Останавливаем поток обучения если он запущен
        if self.training_thread and self.training_thread.isRunning():
            self.results_text.append("🛑 Останавливаю процесс обучения...")
            self.training_thread.stop()  # Сначала мягкая остановка
            self.training_thread.terminate()  # Затем принудительная остановка
            
            # Ждем завершения потока с таймаутом
            if not self.training_thread.wait(5000):  # Увеличиваем таймаут до 5 секунд
                self.results_text.append("⚠️  Принудительное завершение потока...")
                self.training_thread.terminate()
                self.training_thread.wait(2000)  # Дополнительное время для завершения
        
        # Отключаем сигналы потока
        if self.training_thread:
            try:
                self.training_thread.progress_signal.disconnect()
                self.training_thread.finished_signal.disconnect()
                self.training_thread.error_signal.disconnect()
            except:
                pass  # Игнорируем ошибки отключения сигналов
        
        # Закрываем все matplotlib фигуры
        for plot_info in self.distribution_plots:
            if 'figure' in plot_info and plot_info['figure']:
                plt.close(plot_info['figure'])
        for plot_info in self.temporal_plots:
            if 'figure' in plot_info and plot_info['figure']:
                plt.close(plot_info['figure'])
        
        # Принимаем событие закрытия
        event.accept()
        
    def save_analysis_results(self):
        """Сохранение результатов анализа в текстовый файл"""
        if not self.data_loader.check_data_ready():
            self.results_text.setText("❌ Сначала загрузите оба файла данных!")
            return
            
        # Получаем текст результатов
        analysis_text = self.results_text.toPlainText()
        
        if not analysis_text.strip():
            self.results_text.setText("❌ Сначала запустите анализ данных!")
            return
            
        try:
            # Создаем имя файла с текущей датой и временем
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"анализ_{timestamp}.txt"
            
            # Сохраняем результаты в файл
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(analysis_text)
            
            # Показываем сообщение об успешном сохранении
            self.results_text.append(f"\n✅ Результаты анализа сохранены в файл: {filename}")
            
        except Exception as e:
            error_msg = f"❌ Ошибка при сохранении файла: {str(e)}"
            self.results_text.append(error_msg)
            print(f"Error in save_analysis_results: {e}")
        
    def train_model(self):
        """Обучение модели машинного обучения для детекции мошенничества в отдельном потоке"""
        if not self.data_loader.check_data_ready():
            self.results_text.setText("❌ Сначала загрузите оба файла данных!")
            return
        
        # Проверяем, не запущен ли уже процесс обучения
        if self.training_thread and self.training_thread.isRunning():
            self.results_text.append("⚠️  Обучение уже запущено! Дождитесь завершения.")
            return
        
        # Очищаем результаты и показываем сообщение о начале обучения
        self.results_text.clear()
        self.results_text.append("🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ДЕТЕКЦИИ МОШЕННИЧЕСТВА")
        self.results_text.append("=" * 60)
        self.results_text.append("⏳ Подготовка к обучению...")
        self.results_text.append("💡 Обучение будет выполняться в фоновом режиме")
        self.results_text.append("💡 Используются все доступные ядра процессора для ускорения")
        self.results_text.append("💡 Интерфейс останется отзывчивым во время обучения")
        self.results_text.append("")
        
        # Создаем и настраиваем поток обучения
        self.training_thread = ModelTrainingThread(self.data_loader)
        
        # Подключаем сигналы
        self.training_thread.progress_signal.connect(self.update_training_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.error_signal.connect(self.training_error)
        self.training_thread.finished.connect(self.on_training_thread_finished)
        
        # Запускаем обучение
        self.training_thread.start()
    
    def update_training_progress(self, message):
        """Обновление прогресса обучения"""
        self.results_text.append(message)
        # Прокручиваем к концу для отображения новых сообщений
        self.results_text.verticalScrollBar().setValue(
            self.results_text.verticalScrollBar().maximum()
        )
    
    def training_finished(self, result):
        """Обработка завершения обучения"""
        if result.get('success', False):
            self.results_text.append("")
            self.results_text.append("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            self.results_text.append(f"🏆 Лучшая модель: {result['best_model_name']}")
            self.results_text.append(f"💾 Файл модели: {result['model_filename']}")
            if result.get('scaler_filename'):
                self.results_text.append(f"💾 Файл scaler: {result['scaler_filename']}")
            self.results_text.append("")
            self.results_text.append("✅ Теперь вы можете использовать кнопку '🔮 Предсказать' для предсказаний!")
        else:
            self.results_text.append("❌ Обучение завершилось с ошибкой")
        
        # Очищаем ссылку на поток
        self.training_thread = None
    
    def training_error(self, error_message):
        """Обработка ошибки обучения"""
        self.results_text.append("")
        self.results_text.append("❌ ОШИБКА ПРИ ОБУЧЕНИИ:")
        self.results_text.append(error_message)
        self.results_text.append("")
        self.results_text.append("💡 Проверьте:")
        self.results_text.append("   - Загружены ли данные корректно")
        self.results_text.append("   - Содержат ли данные необходимые поля")
        self.results_text.append("   - Достаточно ли памяти для обработки")
        
        # Очищаем ссылку на поток
        self.training_thread = None
    
    def on_training_thread_finished(self):
        """Обработка завершения потока обучения"""
        # Отключаем сигналы
        if self.training_thread:
            try:
                self.training_thread.progress_signal.disconnect()
                self.training_thread.finished_signal.disconnect()
                self.training_thread.error_signal.disconnect()
                self.training_thread.finished.disconnect()
            except:
                pass
        
        # Очищаем ссылку на поток
        self.training_thread = None

    def predict_with_model(self):
        """Загружает сохранённую модель и делает предсказания на новых данных"""
        if not hasattr(self, 'data_loader') or self.data_loader.transactions_df is None:
            self.results_text.append("❌ Сначала загрузите данные для анализа")
            return
            
        self.results_text.clear()
        self.results_text.append("🔮 Предсказание мошенничества с помощью обученной модели\n")
        self.results_text.append("=" * 50)
        
        try:
            import joblib
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            # Ищем сохранённые файлы моделей
            import os
            model_files = [f for f in os.listdir('.') if f.startswith('best_fraud_detection_model_') and f.endswith('.joblib')]
            scaler_files = [f for f in os.listdir('.') if f.startswith('scaler_') and f.endswith('.joblib')]
            
            if not model_files:
                self.results_text.append("❌ Не найдены сохранённые модели!")
                self.results_text.append("Сначала обучите модель с помощью кнопки 'Обучить модель'")
                return
            
            # Загружаем лучшую модель (первую найденную)
            model_filename = model_files[0]
            model = joblib.load(model_filename)
            
            self.results_text.append(f"✅ Загружена модель: {model_filename}")
            
            # Загружаем scaler если есть
            scaler = None
            if scaler_files:
                scaler_filename = scaler_files[0]
                scaler = joblib.load(scaler_filename)
                self.results_text.append(f"✅ Загружен scaler: {scaler_filename}")
            
            # Подготавливаем данные так же, как при обучении
            self.results_text.append("\n📊 Подготавливаю данные для предсказания...")
            
            df = self.data_loader.transactions_df.copy()
            
            # Конвертируем timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6])
            
            # Извлекаем данные из last_hour_activity
            if 'last_hour_activity' in df.columns:
                df['num_transactions_last_hour'] = df['last_hour_activity'].apply(
                    lambda x: x['num_transactions'] if pd.notna(x) else 0)
                df['total_amount_last_hour'] = df['last_hour_activity'].apply(
                    lambda x: x['total_amount'] if pd.notna(x) else 0)
                df['unique_merchants_last_hour'] = df['last_hour_activity'].apply(
                    lambda x: x['unique_merchants'] if pd.notna(x) else 0)
                df['unique_countries_last_hour'] = df['last_hour_activity'].apply(
                    lambda x: x['unique_countries'] if pd.notna(x) else 0)
                df['max_single_amount_last_hour'] = df['last_hour_activity'].apply(
                    lambda x: x['max_single_amount'] if pd.notna(x) else 0)
            
            # Инжиниринг признаков
            # Кодируем категориальные переменные
            categorical_columns = ['vendor_category', 'vendor_type', 'currency', 'country', 
                                   'city_size', 'card_type', 'device', 'channel']
            
            for col in categorical_columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            
            # Создаем новые признаки
            df['amount_log'] = np.log1p(df['amount'])
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
            df['is_low_amount'] = (df['amount'] < df['amount'].quantile(0.05)).astype(int)
            df['activity_intensity'] = df['num_transactions_last_hour'] * df['total_amount_last_hour']
            df['geographic_spread'] = df['unique_countries_last_hour'] * df['unique_merchants_last_hour']
            
            # Выбираем числовые признаки
            numeric_features = [
                'amount', 'amount_log', 'hour', 'hour_sin', 'hour_cos', 'is_weekend',
                'is_outside_home_country', 'is_high_risk_vendor', 'is_card_present',
                'num_transactions_last_hour', 'total_amount_last_hour', 'unique_merchants_last_hour',
                'unique_countries_last_hour', 'max_single_amount_last_hour',
                'is_high_amount', 'is_low_amount', 'activity_intensity', 'geographic_spread'
            ]
            
            # Добавляем закодированные категориальные признаки
            encoded_features = [col for col in df.columns if col.endswith('_encoded')]
            numeric_features.extend(encoded_features)
            numeric_features = [col for col in numeric_features if col in df.columns]
            
            # Подготавливаем данные для предсказания
            df_clean = df[numeric_features].dropna()
            X = df_clean[numeric_features]
            
            self.results_text.append(f"✅ Подготовлено {len(X):,} транзакций для предсказания")
            
            # Делаем предсказания
            self.results_text.append("\n🔮 Выполняю предсказания...")
            
            if scaler is not None:
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
            else:
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]
            
            # Анализируем результаты
            fraud_count = predictions.sum()
            total_count = len(predictions)
            fraud_percentage = (fraud_count / total_count) * 100
            
            self.results_text.append(f"\n📊 РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ:")
            self.results_text.append(f"   Всего транзакций: {total_count:,}")
            self.results_text.append(f"   Предсказано мошеннических: {fraud_count:,}")
            self.results_text.append(f"   Процент мошенничества: {fraud_percentage:.2f}%")
            
            # Статистика по вероятностям
            high_risk_threshold = 0.8
            medium_risk_threshold = 0.5
            
            high_risk_count = (probabilities >= high_risk_threshold).sum()
            medium_risk_count = ((probabilities >= medium_risk_threshold) & (probabilities < high_risk_threshold)).sum()
            low_risk_count = (probabilities < medium_risk_threshold).sum()
            
            self.results_text.append(f"\n🎯 РАСПРЕДЕЛЕНИЕ ПО РИСКАМ:")
            self.results_text.append(f"   Высокий риск (≥80%): {high_risk_count:,} ({high_risk_count/total_count*100:.1f}%)")
            self.results_text.append(f"   Средний риск (50-80%): {medium_risk_count:,} ({medium_risk_count/total_count*100:.1f}%)")
            self.results_text.append(f"   Низкий риск (<50%): {low_risk_count:,} ({low_risk_count/total_count*100:.1f}%)")
            
            # Топ-10 транзакций с наибольшим риском
            risk_df = pd.DataFrame({
                'transaction_id': df_clean.index[:len(probabilities)],
                'amount': df_clean['amount'][:len(probabilities)],
                'fraud_probability': probabilities,
                'predicted_fraud': predictions
            })
            
            top_risky = risk_df.nlargest(10, 'fraud_probability')
            
            self.results_text.append(f"\n🚨 ТОП-10 ТРАНЗАКЦИЙ С ВЫСОКИМ РИСКОМ:")
            for idx, row in top_risky.iterrows():
                status = "🚨 МОШЕННИЧЕСТВО" if row['predicted_fraud'] else "⚠️  ПОДОЗРИТЕЛЬНО"
                self.results_text.append(f"   ID: {row['transaction_id']}, Сумма: {row['amount']:,.0f}₽, "
                                       f"Риск: {row['fraud_probability']:.1%} - {status}")
            
            # Сохраняем результаты в файл
            results_filename = 'fraud_predictions.csv'
            risk_df.to_csv(results_filename, index=False)
            self.results_text.append(f"\n💾 Результаты сохранены в файл: {results_filename}")
            
            # Рекомендации
            self.results_text.append(f"\n💡 РЕКОМЕНДАЦИИ:")
            if fraud_percentage > 5:
                self.results_text.append("   🚨 Высокий уровень мошенничества - требуется немедленное вмешательство")
            elif fraud_percentage > 2:
                self.results_text.append("   ⚠️  Повышенный уровень мошенничества - усилить мониторинг")
            else:
                self.results_text.append("   ✅ Нормальный уровень мошенничества")
            
            if high_risk_count > 0:
                self.results_text.append(f"   🚨 {high_risk_count} транзакций требуют немедленной блокировки")
            
            self.results_text.append(f"\n✅ Предсказание завершено успешно!")
            
        except Exception as e:
            self.results_text.append(f"❌ Ошибка при предсказании: {str(e)}")
            self.results_text.append("Проверьте, что модель была обучена и сохранена корректно.")

def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    
    # Устанавливаем современный стиль
    app.setStyle('Fusion')
    
    # Устанавливаем глобальные стили для приложения
    app.setStyleSheet("""
        QApplication {
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 9pt;
        }
        QMainWindow {
            background-color: #f8f9fa;
        }
        QGroupBox {
            font-weight: bold;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
    """)
    
    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()
    
    # Запускаем приложение
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
