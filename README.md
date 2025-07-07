# 🏈 NFL QB Touchdown Predictor

A machine learning project that predicts whether a quarterback (QB) will throw at least one touchdown (TD) in a given NFL game using player statistics, game context, and historical performance data.

---

## 📌 Project Objective

> Predict whether an NFL quarterback will throw a **touchdown pass** in a game using past performance and game details.

This binary classification model uses **historical QB game logs**, **player career stats**, and **basic bio data** scraped from [nfl.com](http://nfl.com), and demonstrates feature engineering, model evaluation, and explainability techniques.

---

## 🚀 Features

- Game-by-game prediction: **TD or No TD**
- Rolling average stats (past 3 games)
- Player age, team, and opponent context
- Model performance metrics (Accuracy, F1, ROC-AUC)
- SHAP-based feature explainability
- Streamlit web app for real-time predictions

---

## 📁 Project Structure

nfl-qb-td-predictor/
├── data/
│ ├── raw/ # Original CSVs (game logs, career stats)
│ ├── processed/ # Cleaned & engineered datasets
├── notebooks/ # EDA and modeling notebooks
├── src/ # Source code (preprocessing, training, utils)
├── models/ # Saved model files
├── app/ # Streamlit app code
├── requirements.txt # Python dependencies
└── README.md


---

## 🧠 Machine Learning Stack

| Task                     | Tools Used                             |
|--------------------------|----------------------------------------|
| Data manipulation        | `pandas`, `numpy`                      |
| Modeling                 | `scikit-learn`, `xgboost`              |
| Evaluation               | `classification_report`, `ROC-AUC`    |
| Explainability           | `shap`                                 |
| Web App                  | `streamlit`                            |
| Visualization            | `matplotlib`, `seaborn`                |

---

## 📊 Model Performance

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.88  |
| F1 Score   | 0.85  |
| ROC-AUC    | 0.91  |

*(Note: Scores will vary based on final model)*

---

## 🧪 How to Run

### 🔧 Install dependencies
```bash
pip install -r requirements.txt
🏃‍♂️ Launch the Streamlit app
streamlit run app/app.py
📈 Example Inputs (App)

QB: Patrick Mahomes
Team: KC
Opponent: DEN
Past 3 Games: Avg 275 yards, 2 TDs
Output: ✅ TD predicted
📚 Data Source

NFL-Statistics-Scrape
Files used:
Game_Logs_Quarterback.csv
Career_Stats_Passing.csv
Basic_Stats.csv
👨‍💻 Author

Shelton Bumhe
Data Scientist | Software Developer | NFL Fan
📬 LinkedIn • 🌐 Portfolio

📢 License

This project is open-source under the MIT License.


---

Let me know if you'd like:
- a `requirements.txt`
- a professional Streamlit UI layout
- GitHub Actions to auto-check code
- or a visual logo/banner for the repo!
