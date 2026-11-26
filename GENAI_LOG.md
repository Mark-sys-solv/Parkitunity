# GenAI-Log

## 25.11.2026 – Frontend-Umbau Teil 1 von 4: Datenbasis, Priorität, Rangliste

**Was Claude Code gemacht hat**
- `CLAUDE.md` mit unseren Projektregeln angelegt.
- `predict_free()` als einzige Quelle für Prognosewerte (ersetzt `predict()`).
- `build_candidates()` (eine Zeile pro Parkhaus) und neue `rank_garages()` mit vier Prioritäten; die alte Bewertung entfernt.
- Komponente "Was ist dir wichtig?" (`src/ui/priority.py`), App, Karte, Empfehlung und ÖV-Vergleich auf die neue Rangliste umgestellt.
- Platzhalter-Preise, Gewichte und Schwellenwerte in `src/config.py` gesammelt, fehlende Docstrings ergänzt, Tests geschrieben.
- Danach: Regler "Max. Fussweg" und "So viele freie Plätze reichen mir", weil sonst Parkhäuser mit bis zu 87 min Fussweg auf Platz 1 kamen. Die Min-max-Skala entsteht nur noch aus Parkhäusern, die infrage kommen.

**Was wir entschieden haben**
- Alte Rangliste ganz ersetzen, Abfahrtszeit statt nur "jetzt", Preislimit vor der Rangliste, bestehende Fahrzeit-Schätzung behalten, ÖV-Vergleich nutzt das Parkhaus auf Platz 1.

**Was wir selbst angepasst haben**
- _(bitte ergänzen)_

## 25.11.2026 – Frontend-Umbau Teil 2 von 4: Karten "Die besten Parkhäuser für deine Fahrt"

**Was Claude Code gemacht hat**
- Top 3 der Rangliste als Karten (`src/ui/garage_cards.py`), eigenes CSS zentral in `src/ui/styles.py`, Farbschwellen in `src/config.py`.
- Klick auf den Chevron setzt `st.session_state["selected_garage"]`; die Auswahl beim Diagramm "Belegung" nutzt denselben Schlüssel.
- Hinweis, wenn die Live-Daten fehlen und Beispielwerte gezeigt werden, und Meldung bei leerer Rangliste. Tests geschrieben.

**Was wir selbst angepasst haben**
- _(bitte ergänzen)_

## 25.11.2026 – Frontend-Umbau Teil 3 von 4: "Wie kommst du am besten ans Ziel?"

**Was Claude Code gemacht hat**
- Vergleich Auto / ÖV als zwei Karten mit Ersparnis-Zeile (`src/ui/mode_cards.py`), Regler "Parkdauer".
- `compare_modes()` neu: Empfehlung nach Priorität statt über einen Zeitwert in CHF, Kilometerkosten wieder dabei (Platzhalter `COST_PER_KM_CHF`).
- Alte Empfehlungskarte und den einzelnen Google-Maps-Button entfernt, weil sie der neuen Empfehlung widersprochen hätten. Tests angepasst.

**Was wir selbst angepasst haben**
- _(bitte ergänzen)_

## 25.11.2026 – Frontend-Umbau Teil 4 von 4: "Parkhausprognose"

**Was Claude Code gemacht hat**
- Unplausible Prognose (Urania 130 → 194 → 130) untersucht: Ursache war die alte Rückwärtsrechnung entlang der "typischen Woche", nicht Zeitzone, Feiertage oder Feature-Reihenfolge. Regressionstest geschrieben.
- `predict_free()`: keine Abrundung auf 15 min mehr, Kurve beginnt exakt beim Live-Wert, Modell läuft pro Parkhaus nur einmal. Neu: `forecast_curve()`.
- Komponente `src/ui/forecast.py` (Altair) ersetzt das Diagramm "Belegung". Ampel-Schwellen in Karten, Stadtkarte und Prognose vereinheitlicht.

**Was wir selbst angepasst haben**
- _(bitte ergänzen)_
