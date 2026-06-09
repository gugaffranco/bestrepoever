from preprocessing_final import load_data, clean_data, create_features, scale_and_reduce, save_data
from modeling_final import remove_outliers, fit_kmeans, assign_all_labels, save_cluster_assignments
from association_final import basket_operations, get_all_cluster_rules, print_cluster_rules

# Columns to use for clustering
cluster_features = [
    'age',
    'year_first_transaction',
    'progenitores',
    'spend_per_store',
    'fresh_food_ratio',
    'lifetime_spend_groceries',
    'lifetime_spend_alcohol_drinks',
    'lifetime_spend_nonalcohol_drinks',
    'lifetime_spend_hygiene',
    'lifetime_spend_petfood',
    'lifetime_total_distinct_products',
    'percentage_of_products_bought_promotion',
    'promotion_on_stores_hunter',
    'distinct_stores_visited',
    'promotion_guy',
    'healthy_guy',
    'tech_enthusiast',
    'has_loyalty_card',
    'loyal_long_timer',
]

if __name__ == '__main__':
    cust_info = load_data('customer_info.csv')
    cust_info = clean_data(cust_info)
    cust_info = create_features(cust_info)
    X_scaled, scaler = scale_and_reduce(cust_info, cluster_features)
    save_data(X_scaled, 'X_scaled')

    X_scaled_clean, outlier_mask = remove_outliers(X_scaled)
    kmeans, kmeans_labels= fit_kmeans(X_scaled_clean)
    all_labels=assign_all_labels(X_scaled, outlier_mask, kmeans, kmeans_labels)
    save_cluster_assignments(X_scaled, all_labels)

    basket=load_data('customer_basket.csv')
    clusters=load_data('cluster_assignments.csv')
    basket=basket_operations(basket, clusters)
    cluster_rules = get_all_cluster_rules(basket)
    print_cluster_rules(cluster_rules, n=3)