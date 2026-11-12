**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy
**Zajęcia nr / temat:** 4 / Obsługa długich sekwencji: LSTM, GRU i splot 1D
**Autor / grupa:** Adrian Zygmunt GL 5.4
**Data:** 5.11.2026

## 1. Cel ćwiczenia
Celem ćwiczenia było porównanie architektur SimpleRNN, LSTM, GRU oraz Conv1D w zadaniu sumowania wartości z opóźnieniem (tzw. Adding Problem). W stosunku do materiałów bazowych, problem utrudniono wprowadzając opóźnienie – znacznik w kroku $t$ wskazuje wartość do zsumowania znajdującą się w kroku $t-8$. Zbadano również wpływ długości sekwencji (L=50 oraz L=100), a także wpływ rozmiaru stanu ukrytego (dla GRU) oraz pola recepcyjnego (dla Conv1D) na skuteczność modelu.

## 2. Metoda i implementacja

- **Wykorzystane narzędzia i modele:** Python, PyTorch, pandas, matplotlib. Zastosowano klasy modeli: `RNNNet` (SimpleRNN, LSTM, GRU) oraz `Conv1D` (wraz z modyfikacją `VariableConvNet` do testów dylatacji).
- **Najważniejsze fragmenty kodu:** https://github.com/AdrianZygmunt/Uczenie-g-bokie-w-systemach-rozpoznawania-i-przetwarzania-mowy---AdrianZygmunt/tree/main/L4
- **Konfiguracja (parametry, dane):** Uczenie przeprowadzono na zbiorze N=4000 przykładów treningowych i N=800 testowych dla sekwencji L=50 i L=100. Parametry optymalizacji: Adam (lr=5e-3), MSELoss, 30 epok, batch size = 128.
- **Generator danych i weryfikacja:** Przygotowano własną funkcję generującą sekwencje z 8-krokowym opóźnieniem. 

**Ręczna weryfikacja działania generatora na jednym przykładzie:**
* Pozycje znaczników (kroki $t$): [13 18] (indeksy wektora: [12 17])
* Wskazywane pozycje (kroki $t-8$): [ 5 10] (indeksy wektora: [4 9])
* Wartości wskazywane: [0.424 0.383]
* Suma (cel predykcji): 0.8070963025093079

## 3. Wyniki

**Tabela: Zestawienie modeli dla sekwencji z opóźnieniem (L=50 i L=100)**
*(Punktem odniesienia dla predykcji średniej jest błąd MSE na poziomie ok. 0.167)*

| Model | Parametry | L=50 MSE | L=50 Czas (s) | L=100 MSE | L=100 Czas (s) |
|---|---|---|---|---|---|
| SimpleRNN | 4417 | 0.1608 | 3.16 | 0.1607 | 2.14 |
| LSTM | 17473 | 0.1651 | 2.55 | 0.1707 | 2.80 |
| GRU | 13121 | 0.0416 | 3.21 | 0.1499 | 3.02 |
| Conv1D | 49921 | 0.0005 | 3.50 | 0.0041 | 3.07 |


**Wizualizacja: Wpływ długości sekwencji na jakość modeli**

![alt text](Wykres1.png)

Wykres wpływu parametrów stanu ukrytego (GRU) oraz pola recepcyjnego (Conv1D):
![alt text](Wykres2.png)

*   **Wykres (po lewej):** Wpływ rozmiaru stanu ukrytego w GRU (dla L=100). Testowano wartości `[16, 32, 64, 128]`. Błąd MSE w każdym wariancie utrzymywał się na wysokim poziomie (~0.16 - 0.17), zbliżonym do predykcji stałej.
*   **Wykres (po prawej):** Zmiana pola recepcyjnego w Conv1D poprzez dylatacje. Dla małych pól recepcyjnych (RF=15, RF=31) model nie potrafił poprawnie zsumować oddalonych wartości (MSE > 0.13). Przekroczenie pewnego progu szerokości pola (RF=63 oraz RF=127) spowodowało drastyczny spadek błędu (kolejno 0.0076 i 0.0030).

## 4. Analiza i interpretacja
Wprowadzenie opóźnienia między znacznikiem a odczytywaną wartością ($t-8$) drastycznie podniosło poziom trudności w stosunku do wariantu bez opóźnienia. 
1. Sieć **SimpleRNN** całkowicie zawiodła (MSE ~0.16). 
2. Ku zaskoczeniu, przy zachowaniu domyślnych hiperparametrów i 30 epokach uczenia, architektury z bramkami pamięci (**LSTM** i **GRU**) również miały trudności z modelowaniem zadania dla L=100, zwracając wynik na poziomie przewidywania stałej średniej (ok. 0.167). GRU wykazało nieco lepszą zdolność do adaptacji dla krótszych sekwencji (L=50, błąd 0.04).
3. Najlepiej spisała się architektura splotowa **Conv1D**. Wykres analizujący pole recepcyjne wprost uwidacznia, że do skutecznego rozwiązywania tego typu problemów sieć splotowa z uśrednianiem w czasie musi posiadać dostatecznie szerokie "okno" spojrzenia (Receptive Field). Zastosowanie rosnących dylatacji $[1, 2, 4, 8, 16]$ rozszerzyło zasięg komórek splotowych, pozwalając na minimalizację błędu do rewelacyjnego poziomu MSE=0.0041 dla L=100.

## 5. Wnioski
1. Fizyczne opóźnienie wskazywanej zależności obnaża trudności sieci rekurencyjnych (w tym LSTM i GRU) w nauce i propagacji błędu. Rozwiązanie tak sformułowanego problemu przy ich pomocy wymagałoby dłuższego treningu lub staranniejszego doboru hiperparametrów.
2. Architektura **Conv1D** z zastosowaniem dylatacji (tzw. dilated convolutions) jest w stanie wyjątkowo efektywnie wyłapywać długoterminowe, sztywne czasowo zależności, deklasując modele RNN/LSTM w tym konkretnym eksperymencie.
3. Fundamentalnym wymogiem przy konfiguracji sieci splotowej do analizy szeregów czasowych z opóźnieniami jest dobór takiej liczby warstw i kroku dylatacji, aby pole recepcyjne (Receptive Field) sieci mogło obejmować niezbędny kontekst historyczny.
