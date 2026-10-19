**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** Sekwencje i sygnał mowy - Od fali dźwiękowej do reprezentacji dla modelu
**Autor / grupa:** Adrian Zygmunt GL 5.4
**Data:** 15.10.2026

## 1. Cel ćwiczenia
Wygenerowanie i porównanie wykresów dla trzech nagrań: samogłoski, wyrazu i zdania. Jak zmiana parametrów n_fft oraz hop_length wpływa na wygląd wykresów.

## 2. Metoda i implementacja

- Wykorzystane narzędzia i modele.
Python, biblioteki: librosa, numpy, matplotlib
- Najważniejsze fragmenty kodu (lub link do repozytorium/notatnika). 
https://github.com/AdrianZygmunt/Uczenie-g-bokie-w-systemach-rozpoznawania-i-przetwarzania-mowy---AdrianZygmunt/tree/main/L1
- Konfiguracja (parametry, dane).
Parametry domyślne: częstotliwość 16000 Hz, n_fft = 512, hop_length = 160

## 3. Wyniki

Samogłoska ("a"): Wykres jest płaski i poziomy. Dźwięk się nie zmienia, a wartości MFCC są stałe w czasie.

Wyraz ("przetwarzanie"): Widać dużą dynamikę. Wykres skacze, pokazując wyraźne różnice między głośnymi literami (samogłoski) a szumami (np. "sz", "prz").

Zdanie: Wykresy świetnie pokazują przerwy (pauzy) między słowami oraz zmiany głośności (akcenty).

## 4. Analiza i interpretacja

Wpływ n_fft (rozmiar okna analizy): Małe okno (256) daje ostry obraz w czasie, ale mocno rozmazuje częstotliwości. Duże okno (2048) świetnie i ostro pokazuje częstotliwości, ale za to rozmazuje w czasie szybkie zmiany (zlewają się ze sobą).

Wpływ hop_length (krok przesunięcia): Mały krok (80) daje bardzo gładki i dokładny obraz, ale generuje za dużo danych. Bardzo duży krok (640) powoduje utratę szczegółów – wykres gubi krótkie i szybkie dźwięki.

## 5. Wnioski

Przy przetwarzaniu mowy trzeba znaleźć "złoty środek" między dokładnością w czasie a dokładnością częstotliwości. Domyślne wartości (n_fft = 512, hop_length = 160 dla 16 kHz) sprawdzają się najlepiej, bo wyłapują zarówno stabilne dźwięki, jak i szybkie zmiany. MFCC to z kolei bardzo dobry sposób na kompresję – zostawia w danych tylko to, co jest najważniejsze do rozpoznania głosu.