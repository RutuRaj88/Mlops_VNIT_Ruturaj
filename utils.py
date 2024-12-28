from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2):
    '''
    :param df: DataFrame
    :return: train_df, test_df
    '''
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    target = 'Species'
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.3, shuffle=True
    )
    return X_train, X_test, y_train, y_test
