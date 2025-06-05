**NBS-ATDA** is a data analysis and visualization project focused on crime data. It utilizes Python libraries such as pandas, matplotlib, and scikit-learn to process and analyze the dataset. The project includes a Streamlit dashboard for interactive data exploration.

## 📁 Project Structure

- `crime_dashboard.py`: Main Streamlit application for data visualization.
- `DA.ipynb`: Jupyter notebook containing exploratory data analysis (EDA).
- `cleaned_crime_data.csv`: Preprocessed crime dataset used for analysis.
- `requirements.txt`: List of Python dependencies required to run the project.
- `.devcontainer/`: Configuration files for development container setup.

## 🚀 Getting Started

To run the Streamlit dashboard locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vasanthakumar13112000/NBS-ATDA.git
   cd NBS-ATDA
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run crime_dashboard.py
   ```

   The dashboard will be accessible at `http://localhost:8501/` in your web browser.

## 🧪 Exploratory Data Analysis

The `DA.ipynb` notebook provides a comprehensive exploratory data analysis of the crime dataset, including:

- Data cleaning and preprocessing steps.
- Statistical summaries and distributions.
- Visualizations to identify trends and patterns.
- Preliminary insights and observations.

## 📊 Streamlit Dashboard Features

The Streamlit dashboard (`crime_dashboard.py`) offers interactive visualizations and insights, including:

- Crime trends over time.
- Geographical distribution of crimes.
- Crime types and their frequencies.
- Filtering options to explore specific subsets of data.

## 📦 Dependencies

The project relies on the following Python libraries:

- pandas
- matplotlib
- scikit-learn
- streamlit

Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

## 💡 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to fork the repository and submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
