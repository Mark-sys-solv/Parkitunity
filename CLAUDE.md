# Regeln für Parkitunity

Parkitunity ist eine Streamlit-App, die in Zürich ein Parkhaus empfiehlt und Auto mit ÖV vergleicht.

## Stack
- Python, Streamlit, pandas, numpy, requests. Keine neuen Libraries ohne Rückfrage.
- Ok, weil sie mit Streamlit oder scikit-learn kommen: Altair, pydeck (Karte), joblib (Modell speichern).
- scikit-learn wird nur im ML-Teil (`src/ml/`) benutzt.

## Code
- UI-Texte auf Deutsch, Schweizer Schreibweise ("ss" statt "ß"). Code, Docstrings und Kommentare auf Englisch, einfach gehalten.
- Jede Funktion hat einen Docstring, nicht-triviale Stellen werden kommentiert (wird benotet).
- Single Source of Truth: Prognosewerte kommen immer aus `predict_free()` (`src/ml/predict.py`), nie doppelt berechnet.
- Konstanten (Gewichte, Schwellenwerte, Platzhalter-Preise) gehören in `src/config.py`.
- KI-generierter Code wird im Quelltext mit einem Kommentar zitiert.

## Nach jeder Aufgabe
- Kurzer Eintrag in `GENAI_LOG.md`: Datum, was Claude Code gemacht hat, was wir selbst angepasst haben.
- Tests laufen lassen: `pytest`.
