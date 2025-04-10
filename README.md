
# 🧠 Customer Behavior Using Unsupervised Learning

Leverage unsupervised machine learning to analyze and visualize customer segmentation based on mall shopping behavior. This project helps uncover hidden patterns in customer demographics and spending habits using clustering techniques.

---

## 📌 Overview

This project uses **K-Means Clustering** to segment mall customers into distinct groups based on attributes like age, annual income, and spending score. The goal is to assist businesses in understanding different customer profiles to target marketing strategies more effectively.

---

## 🛠 Project Structure

```
Mall_UCS_App/
│
├── app.py                      
├── requirements.txt              
├── mall_customers.csv           
├── Unsupervised_Clustering_Solution.ipynb  
│
├── assets/
│   └── style.css                 
│
├── utils/
│   ├── __init__.py
│   ├── cluster_model.py         
│   └── load_data.py             
│
└── pages/
    └── Overview.py             
```

---

## 🚀 How to Run

Follow these steps to run the Streamlit app locally:

### 🌐 Step 1: Clone the Repository

```bash
git clone https://github.com/arjunvk007/Mall_UCS_App.git
cd Mall_UCS_App
```

### 🌐 Step 2: Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🎯 Features

- Interactive customer clustering using K-Means
- Customizable number of clusters
- Beautiful UI with custom CSS

---

## 🗃 Data

The dataset used is:

- **File:** `mall_customers.csv`
- **Columns:** `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

---

## 📈 Technologies Used

- **Python** 
- **Streamlit** 
- **Pandas**, **NumPy** 
- **Scikit-learn** 
- **Matplotlib**, **Seaborn** 
- **CSS** 

---

## 🌩 Deployment

You can deploy this Streamlit app on:

- **Streamlit Cloud**
- **Heroku**
- **Render**
- **Localhost for testing**

---

## Link

https://mallucsapp-arjunvk.streamlit.app/

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 🙌 Contribution

Feel free to fork this repository and make improvements or adjustments. Pull requests and contributions are always welcome!

---

## ✉️ Contact

- **Author:** Arjun Vannathan Kandy  
- **GitHub:** [[arjunvk007](https://github.com/arjunvk007)]
