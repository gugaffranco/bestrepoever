import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_data(path):
    cust_info=pd.read_csv(path)
    cust_info=cust_info.set_index('customer_id')
    print(f'Data loaded: {path}')
    return cust_info


def clean_data(cust_info):
    # First parse dates and convert to age
    cust_info['customer_birthdate'] = pd.to_datetime(cust_info['customer_birthdate'], format='%m/%d/%Y %I:%M %p')
    reference_date = pd.Timestamp.today()
    cust_info['age'] = (reference_date - cust_info['customer_birthdate']).dt.days // 365

    # Then impute the resulting age with median (no rows dropped)
    cust_info['age'] = cust_info['age'].fillna(cust_info['age'].median())
    cust_info = cust_info.drop(columns=['customer_birthdate'])

    cust_info['has_loyalty_card'] = cust_info['loyalty_card_number'].notna().astype(int)
    cust_info = cust_info.drop(columns=['loyalty_card_number'])

    cols_to_impute = [
        'kids_home', 'teens_home', 'number_complaints',
        'distinct_stores_visited', 'lifetime_spend_electronics', 'typical_hour',
        'lifetime_spend_vegetables', 'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat',
        'lifetime_spend_fish', 'lifetime_spend_hygiene', 'lifetime_spend_videogames',
        'lifetime_spend_petfood', 'percentage_of_products_bought_promotion',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_groceries',
        'lifetime_total_distinct_products', 'year_first_transaction'
    ]

    imputer = SimpleImputer(strategy='median')
    cust_info[cols_to_impute] = imputer.fit_transform(cust_info[cols_to_impute])

    cust_info['education_level'] = cust_info['customer_name'].str.extract(r'^(Bsc|Msc|Phd)\.').fillna('None')
    cust_info['customer_name'] = cust_info['customer_name'].str.replace(r'^(Bsc|Msc|Phd)\.\s*', '', regex=True)

    # I turned the negative values into positive ones
    cust_info['percentage_of_products_bought_promotion'] = cust_info['percentage_of_products_bought_promotion'].where(
        cust_info['percentage_of_products_bought_promotion'] >= 0, cust_info['percentage_of_products_bought_promotion'] * -1)

    # Cap future dates to the most recent plausible year
    cust_info['year_first_transaction'] = cust_info['year_first_transaction'].clip(upper=2026)

    # Cap pre-2000 to 2000 (earliest plausible year)
    cust_info['year_first_transaction'] = cust_info['year_first_transaction'].clip(lower=2000)
    
    print('\nData cleaned')

    return cust_info


def create_features(cust_info):
    reference_date = pd.Timestamp.today()
    cust_info['fresh_food_ratio'] = (cust_info['lifetime_spend_vegetables'] + cust_info['lifetime_spend_meat'] + cust_info['lifetime_spend_fish']) / cust_info['lifetime_spend_groceries']
    cust_info['spend_per_store'] = cust_info['lifetime_spend_groceries'] / cust_info['distinct_stores_visited']
    cust_info['promotion_on_stores_hunter'] = cust_info['percentage_of_products_bought_promotion'] * cust_info['distinct_stores_visited']
    cust_info['promotion_guy'] = cust_info['lifetime_spend_groceries'] * cust_info['percentage_of_products_bought_promotion']
    cust_info['progenitores'] = cust_info['kids_home'] + cust_info['teens_home']
    cust_info['family_oriented_shopper'] = (cust_info['lifetime_spend_groceries'] + cust_info['lifetime_spend_meat'] + cust_info['lifetime_spend_vegetables'])/cust_info['lifetime_total_distinct_products']
    cust_info['tech_enthusiast'] = (cust_info['lifetime_spend_electronics'] + cust_info['lifetime_spend_videogames'])/cust_info['lifetime_spend_groceries']
    cust_info['healthy_guy'] = (cust_info['lifetime_spend_vegetables'] + cust_info['lifetime_spend_fish'])/cust_info['lifetime_spend_groceries']
    cust_info['loyal_long_timer'] = (reference_date.year - cust_info['year_first_transaction']) * cust_info['has_loyalty_card']

    cust_info = cust_info.replace([np.inf, -np.inf], 0) #sets all infinite values to 0
    print('Features created')

    return cust_info


def scale_and_reduce(cust_info, cluster_features):
    
    X = cust_info[cluster_features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=cluster_features,
        index=cust_info.index
    )

    print('Data scaled')

    return X_scaled, scaler


def save_data(file, name):
        file.to_csv(f'{name}.csv', index=True)
        print('\nData saved')


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
     cust_info=load_data('customer_info.csv')
     cust_info=clean_data(cust_info)
     cust_info=create_features(cust_info)
     X_scaled, scaler = scale_and_reduce(cust_info, cluster_features)
     save_data(X_scaled, 'X_scaled')