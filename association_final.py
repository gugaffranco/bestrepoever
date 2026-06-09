import ast
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from preprocessing_final import load_data

def basket_operations(basket, clusters):
    basket = basket.merge(clusters, on='customer_id')
    basket['list_of_goods'] = basket['list_of_goods'].apply(ast.literal_eval)
    basket = basket.dropna(subset=['cluster'])
    basket['cluster'] = basket['cluster'].astype(int)
    print('\nBasket merged and cleaned\n')
    return basket

def get_association_rules(transactions, min_support=0.02, min_confidence=0.2):
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        print("No frequent itemsets found")
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    return rules.sort_values('lift', ascending=False)

def get_cluster_rules(basket, cluster_id):
    transactions = basket[basket['cluster'] == cluster_id]['list_of_goods'].tolist()
    rules = get_association_rules(transactions)
    return rules

def get_all_cluster_rules(basket):
    cluster_rules = {}
    for cluster_id in sorted(basket['cluster'].unique()):
        print(f'Getting the clusters rules for cluster {cluster_id}')
        cluster_rules[cluster_id] = get_cluster_rules(basket, cluster_id)
    return cluster_rules

def print_cluster_rules(cluster_rules, n=5):
    for cluster_id, rules in cluster_rules.items():
        print(f"\n{'='*50}")
        print(f"CLUSTER {cluster_id} - Top Rules by Lift")
        print(f"{'='*50}")
        if not rules.empty:
            rules = rules.copy()
            rules['pair'] = rules.apply(
                lambda row: tuple(sorted([str(sorted(row['antecedents'])), str(sorted(row['consequents']))])), axis=1
            )
            unique_rules = rules.drop_duplicates(subset='pair')
            print(unique_rules.nlargest(n, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            print("No rules found")

if __name__ == '__main__':
    basket=load_data('customer_basket.csv')
    clusters=load_data('cluster_assignments.csv')
    basket=basket_operations(basket, clusters)
    cluster_rules = get_all_cluster_rules(basket)
    print_cluster_rules(cluster_rules, n=3)