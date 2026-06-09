import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Brand colour used throughout ──────────────────────────────────────────────
PRIMARY_COLOR = "#25a890"


# ── Generic helpers ───────────────────────────────────────────────────────────

def plot_histogram(series: pd.Series, title: str, xlabel: str, bins: int = 10) -> None:
    """Histogram for any numeric series."""
    sns.histplot(data=series, color=PRIMARY_COLOR, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_boxplot(series: pd.Series, title: str, ylabel: str) -> None:
    """Vertical boxplot for any numeric series."""
    sns.boxplot(y=series, color=PRIMARY_COLOR)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_countplot(series: pd.Series, title: str, xlabel: str,
                   palette: dict | None = None) -> None:
    """Count-plot for a categorical / low-cardinality series."""
    sns.countplot(x=series, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ── Per-variable plots ────────────────────────────────────────────────────────

def plot_gender_distribution(df: pd.DataFrame) -> None:
    palette = {"male": "lightblue", "female": "lightpink"}
    plot_countplot(df["customer_gender"], "Customer Gender Distribution",
                   "Gender", palette=palette)


def plot_birth_year_distribution(df: pd.DataFrame) -> None:
    birth_year = pd.to_datetime(df["customer_birthdate"]).dt.year
    plot_histogram(birth_year, "Customer Year of Birth Distribution", "Year of Birth", bins=31)
    plot_boxplot(birth_year, "Boxplot of Customer Year of Birth", "Year of Birth")


def plot_kids_home(df: pd.DataFrame) -> None:
    plot_histogram(df["kids_home"], "Customer Kids at Home Distribution", "Kids at Home", bins=8)
    plot_boxplot(df["kids_home"], "Boxplot of Customer Kids at Home", "Kids at Home")


def plot_teens_home(df: pd.DataFrame) -> None:
    plot_histogram(df["teens_home"], "Customer Teens at Home Distribution", "Teens at Home", bins=6)
    plot_boxplot(df["teens_home"], "Boxplot of Customer Teens at Home", "Teens at Home")


def plot_number_complaints(df: pd.DataFrame) -> None:
    plot_histogram(df["number_complaints"], "Customer Number of Complaints Distribution",
                   "Number of Complaints", bins=7)
    plot_boxplot(df["number_complaints"], "Boxplot of Customer Complaints",
                 "Number of Complaints")


def plot_distinct_stores_visited(df: pd.DataFrame) -> None:
    plot_histogram(df["distinct_stores_visited"],
                   "Customer Distinct Stores Visited Distribution",
                   "Distinct Stores Visited", bins=10)
    plot_boxplot(df["distinct_stores_visited"],
                 "Boxplot of Customer Distinct Stores Visited",
                 "Distinct Stores Visited")


def plot_lifetime_spend_groceries(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_groceries"],
                   "Customer Lifetime Spend on Groceries Distribution",
                   "Lifetime Spend on Groceries", bins=10)
    plot_boxplot(df["lifetime_spend_groceries"],
                 "Boxplot of Customer Lifetime Spend on Groceries",
                 "Lifetime Spend on Groceries")


def plot_lifetime_spend_electronics(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_electronics"],
                   "Customer Lifetime Spend on Electronics Distribution",
                   "Lifetime Spend on Electronics", bins=5)
    plot_boxplot(df["lifetime_spend_electronics"],
                 "Boxplot of Customer Lifetime Spend on Electronics",
                 "Lifetime Spend on Electronics")


def plot_typical_hour(df: pd.DataFrame) -> None:
    plot_histogram(df["typical_hour"],
                   "Customer Typical Hour of Visit Distribution",
                   "Typical Hour of Visit", bins=17)
    plot_boxplot(df["typical_hour"], "Boxplot of Customer Typical Hour",
                 "Typical Hour of Visit")


def plot_lifetime_spend_vegetables(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_vegetables"],
                   "Customer Lifetime Spend on Vegetables Distribution",
                   "Lifetime Spend on Vegetables", bins=10)
    plot_boxplot(df["lifetime_spend_vegetables"],
                 "Boxplot of Customer Lifetime Spend on Vegetables",
                 "Lifetime Spend on Vegetables")


def plot_lifetime_spend_nonalcohol_drinks(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_nonalcohol_drinks"],
                   "Customer Lifetime Spend on Non-Alcohol Drinks Distribution",
                   "Lifetime Spend on Non-Alcohol Drinks", bins=10)
    plot_boxplot(df["lifetime_spend_nonalcohol_drinks"],
                 "Boxplot of Customer Lifetime Spend on Non-Alcohol Drinks",
                 "Lifetime Spend on Non-Alcohol Drinks")


def plot_lifetime_spend_alcohol_drinks(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_alcohol_drinks"],
                   "Customer Lifetime Spend on Alcohol Drinks Distribution",
                   "Lifetime Spend on Alcohol Drinks", bins=10)
    plot_boxplot(df["lifetime_spend_alcohol_drinks"],
                 "Boxplot of Customer Lifetime Spend on Alcohol Drinks",
                 "Lifetime Spend on Alcohol Drinks")


def plot_lifetime_spend_meat(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_meat"],
                   "Customer Lifetime Spend on Meat Distribution",
                   "Lifetime Spend on Meat", bins=10)
    plot_boxplot(df["lifetime_spend_meat"],
                 "Boxplot of Customer Lifetime Spend on Meat",
                 "Lifetime Spend on Meat")


def plot_lifetime_spend_fish(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_fish"],
                   "Customer Lifetime Spend on Fish Distribution",
                   "Lifetime Spend on Fish", bins=10)
    plot_boxplot(df["lifetime_spend_fish"],
                 "Boxplot of Customer Lifetime Spend on Fish",
                 "Lifetime Spend on Fish")


def plot_lifetime_spend_hygiene(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_hygiene"],
                   "Customer Lifetime Spend on Hygiene Distribution",
                   "Lifetime Spend on Hygiene", bins=10)
    plot_boxplot(df["lifetime_spend_hygiene"],
                 "Boxplot of Customer Lifetime Spend on Hygiene Products",
                 "Lifetime Spend on Hygiene Products")


def plot_lifetime_spend_videogames(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_videogames"],
                   "Customer Lifetime Spend on Videogames Distribution",
                   "Lifetime Spend on Videogames", bins=20)
    plot_boxplot(df["lifetime_spend_videogames"],
                 "Boxplot of Customer Lifetime Spend on Videogames",
                 "Lifetime Spend on Videogames")


def plot_lifetime_spend_petfood(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_spend_petfood"],
                   "Customer Lifetime Spend on Pet Food Distribution",
                   "Lifetime Spend on Pet Food", bins=12)
    plot_boxplot(df["lifetime_spend_petfood"],
                 "Boxplot of Customer Lifetime Spend on Pet Food",
                 "Lifetime Spend on Pet Food")


def plot_lifetime_total_distinct_products(df: pd.DataFrame) -> None:
    plot_histogram(df["lifetime_total_distinct_products"],
                   "Customer Lifetime Total Distinct Products Distribution",
                   "Lifetime Total Distinct Products", bins=10)
    plot_boxplot(df["lifetime_total_distinct_products"],
                 "Boxplot of Customer Lifetime Total Distinct Products",
                 "Lifetime Total Distinct Products")


def plot_percentage_bought_promotion(df: pd.DataFrame) -> None:
    plot_histogram(df["percentage_of_products_bought_promotion"],
                   "Customer Lifetime Spend on Promotion Distribution",
                   "Lifetime Spend on Promotion", bins=10)
    plot_boxplot(df["percentage_of_products_bought_promotion"],
                 "Boxplot of Customer Percentage of Products Bought on Promotion",
                 "Percentage of Products Bought on Promotion")


def plot_year_first_transaction(df: pd.DataFrame) -> None:
    plot_histogram(df["year_first_transaction"],
                   "Customer First Transaction Year Distribution",
                   "Year of First Transaction", bins=32)
    plot_boxplot(df["year_first_transaction"],
                 "Boxplot of Customer Year of First Transaction",
                 "Year of First Transaction")


def plot_loyalty_card_distribution(df: pd.DataFrame) -> None:
    filled = df["loyalty_card_number"].fillna(0).astype(int).astype(str)
    palette = {"1": "lightcoral", "0": "lightgreen"}
    plot_countplot(filled, "Customer Loyalty Card Distribution",
                   "Has Loyalty Card", palette=palette)


def plot_geographic_density(df: pd.DataFrame) -> None:
    """KDE density plot of customer locations."""
    sns.kdeplot(data=df, x="longitude", y="latitude",
                fill=True, color="#0c7a33")
    plt.title("Customer Geographic Density")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()


# ── Correlation plots ─────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Full numerical correlation matrix, mirroring the notebook logic."""
    corr_df = df.copy()
    corr_df["age"] = pd.Timestamp.now().year - pd.to_datetime(
        corr_df["customer_birthdate"]).dt.year
    corr_df["degree"] = corr_df["customer_name"].str.extract(
        r"(PhD|Master|Bachelor|Dr)", expand=False)

    le = LabelEncoder()
    corr_df["customer_gender_encoded"] = le.fit_transform(
        corr_df["customer_gender"].astype(str))
    corr_df["degree"] = le.fit_transform(corr_df["degree"].astype(str))

    corr_cols = [
        "age", "customer_gender_encoded", "degree", "kids_home", "teens_home",
        "number_complaints", "distinct_stores_visited", "lifetime_spend_groceries",
        "lifetime_spend_electronics", "typical_hour", "lifetime_spend_vegetables",
        "lifetime_spend_nonalcohol_drinks", "lifetime_spend_alcohol_drinks",
        "lifetime_spend_meat", "lifetime_spend_fish", "lifetime_spend_hygiene",
        "lifetime_spend_videogames", "lifetime_spend_petfood",
        "lifetime_total_distinct_products", "percentage_of_products_bought_promotion",
        "year_first_transaction", "loyalty_card_number", "latitude", "longitude",
    ]

    corr_matrix = corr_df[corr_cols].corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Greens",
                center=0, linewidths=0.5, annot_kws={"size": 7})
    plt.title("Correlation Matrix of All Customer Variables")
    plt.tight_layout()
    plt.show()


# ── Orchestrators ─────────────────────────────────────────────────────────────

def plot_all_variable_distributions(df: pd.DataFrame) -> None:
    """Run every per-variable histogram + boxplot in sequence."""
    plot_gender_distribution(df)
    plot_birth_year_distribution(df)
    plot_kids_home(df)
    plot_teens_home(df)
    plot_number_complaints(df)
    plot_distinct_stores_visited(df)
    plot_lifetime_spend_groceries(df)
    plot_lifetime_spend_electronics(df)
    plot_typical_hour(df)
    plot_lifetime_spend_vegetables(df)
    plot_lifetime_spend_nonalcohol_drinks(df)
    plot_lifetime_spend_alcohol_drinks(df)
    plot_lifetime_spend_meat(df)
    plot_lifetime_spend_fish(df)
    plot_lifetime_spend_hygiene(df)
    plot_lifetime_spend_videogames(df)
    plot_lifetime_spend_petfood(df)
    plot_lifetime_total_distinct_products(df)
    plot_percentage_bought_promotion(df)
    plot_year_first_transaction(df)
    plot_loyalty_card_distribution(df)
    plot_geographic_density(df)


def run_all_eda_plots(df: pd.DataFrame) -> None:
    """Full EDA visual suite: distributions + correlation heatmap."""
    plot_all_variable_distributions(df)
    plot_correlation_heatmap(df)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    cust_info = pd.read_csv("customer_info.csv", index_col=0)

    run_all_eda_plots(cust_info)