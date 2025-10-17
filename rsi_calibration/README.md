<!-- README for the rsi_calibration package -->

# RSI Calibration Workflows

Pakiet `rsi_calibration` zawiera eksperymentalne pipeline’y związane z kalibracją wskaźnika RSI oraz jego użyciem w klasyfikacji drop/flat/rise. Poniżej krótka instrukcja pracy i opis dostępnych case’ów.

## 1. Przygotowanie danych (obowiązkowe przed uruchomieniem case’ów)

1. **Pobierz dane TSLA z Yahoo Finance**  
   Uruchom skrypt:
   ```bash
   python rsi_calibration/fetch_tsla_data.py
   ```
   Domyślnie zapisze on 10 lat dziennych notowań do pliku `rsi_calibration/data/tsla_yf_dataset.csv`.  
   Jeśli chcesz wskazać inną lokalizację:
   ```bash
   python -c "from rsi_calibration.fetch_tsla_data import fetch_tsla; fetch_tsla(output_path='ścieżka/do/pliku.csv')"
   ```

2. **Opcjonalnie**: Zamień ten plik na wejście do innych pipeline’ów (np. kopiując do `data/ml_input/ml_dataset.csv`), jeżeli chcesz pracować na świeżym zakresie dat. W przeciwnym razie standardowe case’y użyją aktualnego datasetu z `data/ml_input`.

## 2. Dostępne case’y

### Case 1: `case_1_tsla_rsi_baseline.py`
- Buduje prosty zestaw cech oparty wyłącznie na RSI (domyślnie 14 okresów).
- Domyślnie korzysta z pliku `rsi_calibration/data/tsla_yf_dataset.csv`; możesz wskazać inny dataset flagą `--dataset`.
- Tworzy etykiety drop/flat/rise według neutralnej strefy (`DEFAULT_NEUTRAL_BAND`).
- Dzieli dane chronologicznie na train/test (80/20).
- Trenuje 10 klasyfikatorów (LR, SVC, KNN, GaussianNB, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, DecisionTree, MLP) na identycznym podziale.
- Zapisuje:
  - Migawkę datasetu (`*_rsi_dataset.csv`) z oznaczeniem, które rekordy trafiły do treningu lub testu.
  - Metryki klasyfikacyjne per klasa (`*_metrics.csv`): Brier score, log loss, ROC AUC (jeśli możliwe), mutual information, accuracy, macro precision/recall/F1.
  - Jeśli dostępna jest biblioteka `great_tables` (oraz lokalnie działa renderowanie WebDrivera), generowana jest tabela PNG z wynikami.
- Uwaga: niektóre metryki (np. AUC) mogą być oznaczone jako `NaN`, gdy w wybranym wycinku testowym występuje tylko jedna etykieta.

### Case 2: `case_2_tsla_target_distribution.py`
- Analizuje dystrybucję etykiet drop/flat/rise dla tego samego feedu (RSI bazowego).
- Tworzy podsumowanie liczby obserwacji i udziału procentowego na klasę (`*_target_distribution.csv`).
- Renderuje tabelę (PNG + HTML) z nagłówkiem i uwagami interpretacyjnymi (wymaga `great_tables`).
- Generuje wykres kołowy (`*_target_distribution_pie.png`) w stylu Seaborn pokazujący udział klas.
- Parametry wejściowe (ticker, neutral band, dataset) są spójne z Case 1, dzięki czemu porównania pozostają proste.

### Case 3: `case_3_tsla_rsi_multihorizon.py`
- Porównuje skuteczność RSI-14 w klasyfikacji drop/flat/rise dla kilku horyzontów przyszłej stopy zwrotu (domyślnie 5/14/21 dni).
- Zachowuje chronologiczny podział train/test, wykorzystując ten sam zestaw modeli co Case 1.
- Zapisuje szczegółowe metryki per horyzont (`*_metrics.csv`), migawkę danych (`*_dataset_snapshot.csv`) oraz zestawienie accuracy (`*_horizon_summary.csv` + tabela PNG/HTML).

### Case 4: `case_4_tsla_rsi_heatmap.py`
- Buduje mapę ciepła pokazującą średnią przyszłą stopę zwrotu w zależności od przedziału RSI i horyzontu (domyślnie 5/14/21 dni, z naciskiem na 21-dniowy target).
- Dane wyjściowe obejmują pivot (`*_heatmap_data.csv`), liczbę obserwacji na koszyk (`*_bucket_counts.csv`), tabelę PNG/HTML i wykres heatmap PNG.
- Umożliwia szybkie sprawdzenie, czy np. RSI < 25 przekłada się na dodatnie wyniki po 21 dniach.

### (Opcjonalnie) `case_1_metrics_table_demo.py`
- Pomocniczy renderer, który na podstawie pliku `*_metrics.csv` generuje dodatkową tabelę PNG+HTML z nagłówkiem oraz notatkami interpretacyjnymi.
- Dodatkowo zapisuje słownik kolumn (`*_metrics_glossary.csv/html`) opisujący znaczenie każdej metryki.
- Przykładowe uruchomienie:
  ```bash
  python rsi_calibration/case_1_metrics_table_demo.py \
      --metrics rsi_calibration/outputs/case_1/tsla_case_1_metrics.csv
  ```

## 3. Uwagi praktyczne

- Wszystkie case’y utrzymują stały podział train/test, aby łatwo porównywać wyniki po modyfikacjach.
- W razie rozbudowy pakietu warto tworzyć kolejne case’y w stylu `case_2_...`, `case_3_...` i dodać sekcję w tym README, co dany eksperyment robi.
- Jeśli chcesz eksperymentować z innymi modelami lub metrykami, pamiętaj, by nie mieszać rekordów między train/test (najlepiej zmieniać tylko pipeline, zostawiając podział danych bez zmian).

Miłej kalibracji!
