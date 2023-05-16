import pandas as pd
from DecisionTree.Tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def clean_dataset(dataset):

    # remove duplicates
    reduced_dataset = dataset.drop_duplicates()

    try:
        # replace missing data with zeros
        reduced_dataset.fillna(value=0, inplace=True)
        return reduced_dataset
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return reduced_dataset.DataFrame()  # Return an empty DataFrame if an error occurs


def print_heatmap(numeric_dataset):
    corr = numeric_dataset.corr(method='pearson')
    cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
    sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)
    plt.show()


def print_histogram(numeric_dataset):
    columns = numeric_dataset.columns
    fig, axes = plt.subplots(1, len(columns), figsize=(18, 6), sharey='all')
    for i in range(len(columns)):
        sns.histplot(numeric_dataset, ax=axes[i], x=columns[i], kde=True)
    plt.show()


def run():
    dataset = pd.read_csv('Bank_dataset.csv', sep=";")
    # print(dataset)
    cleaned_dataset = clean_dataset(dataset)
    # print(cleaned_dataset)

    # # Select only the numeric columns
    # numeric_dataset = cleaned_dataset.select_dtypes(include='number')
    # # numeric_dataset = pd.concat([numeric_dataset, output_column], axis=1)
    # # print(numeric_dataset)
    # # numeric_data = np.array(numeric_dataset)

    # print_heatmap(numeric_dataset)
    # print_histogram(numeric_dataset)
    # split data
    x = cleaned_dataset.iloc[:, :-1].values
    y = cleaned_dataset.iloc[:,-1].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=.2, random_state=41)
    # print(x_train, y_train)
    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(x_train, y_train)
    classifier.print_tree()
    # print(len(y_test))
    # y_predict = naive_bayes_categorical(train, x_test, output_column.name)
    #
    # # print(y_test)
    # # print(y_predict)
    # # print(len(y_predict))
    # print(confusion_matrix(y_test, y_predict))
    # print(f1_score(y_test, y_predict, pos_label='no'))

    y_predict = classifier.predict(x_test)

    from sklearn.metrics import accuracy_score
    print('Accuracy : {:.2f} %'.format(accuracy_score(y_test, y_predict)*100))


if __name__ == "__main__":
    run()
