import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path = "thyroidDF.csv"

uselessData = ['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured',
               'patient_id', 'referral_source']
diagnoses = {'-': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 2, 'J': 2, 'K': 2, 'L': 2,
             'M': 2, 'N': 2, 'O': 2, 'P': 2, 'Q': 2, 'R': 2, 'S': 2, 'T': 2}
feature_cols = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick',
                'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
                'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'TT4', 'T4U', 'FTI']


class ProcessData:
    def load_data(self):
        return pd.read_csv(path)

    def process_data(self, dataset: pd.DataFrame):
        dataset.drop(labels=uselessData, axis=1, inplace=True)
        dataset['target'] = dataset['target'].map(diagnoses)
        dataset = self.replace_boolean(dataset)
        dataset = dataset.fillna(value=0)

        # Print the first few rows of the processed dataset
        print(dataset.head())

        # Print unique values in the 'target' column
        print("Unique values in 'target' column:", dataset['target'].unique())

        return dataset

    def split_to_train_and_test(self, dataset: pd.DataFrame):
        # Input data
        x = dataset.loc[:, feature_cols]
        # Output data
        y = dataset['target']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=231056)

        return X_train, X_test, y_train, y_test

    def replace_boolean(self, dataset: pd.DataFrame) -> pd.DataFrame:
        boolean_columns = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
                           'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
                           'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']

        # Replace sex with 0/1 values
        dataset['sex'] = dataset['sex'].replace({'F': 1, 'M': 0})

        # Replace rest values with 0/1
        dataset[boolean_columns] = dataset[boolean_columns].replace({'t': 1, 'f': 0})

        return dataset

    def check_thyroid_status(self, model, scaler):
        age = float(input("Age: "))
        sex = input("Sex (M/F): ").upper()
        on_thyroxine = input("Are you on thyroxine? (y/n): ").lower()
        query_on_thyroxine = input("Do you have queries about thyroxine? (y/n): ").lower()
        on_antithyroid_meds = input("Are you on antithyroid medications? (y/n): ").lower()
        sick = input("Are you sick? (y/n): ").lower()
        pregnant = input("Are you pregnant? (y/n): ").lower()
        thyroid_surgery = input("Have you undergone thyroid surgery? (y/n): ").lower()
        I131_treatment = input("Have you been treated with I131? (y/n): ").lower()
        query_hypothyroid = input("Do you have doubts about hypothyroidism? (y/n): ").lower()
        query_hyperthyroid = input("Do you have doubts about hyperthyroidism? (y/n): ").lower()
        lithium = input("Are you taking lithium? (y/n): ").lower()
        tumor = input("Do you have a thyroid tumor? (y/n): ").lower()
        hypopituitary = input("Do you have hypopituitarism? (y/n): ").lower()
        psych = input("Do you have mental health issues? (y/n): ").lower()
        TSH = float(input("Enter TSH value: "))
        TT4 = float(input("Enter TT4 value: "))
        T4U = float(input("Enter T4U value: "))
        FTI = float(input("Enter FTI value: "))

        new_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'on_thyroxine': [on_thyroxine],
            'query_on_thyroxine': [query_on_thyroxine],
            'on_antithyroid_meds': [on_antithyroid_meds],
            'sick': [sick],
            'pregnant': [pregnant],
            'thyroid_surgery': [thyroid_surgery],
            'I131_treatment': [I131_treatment],
            'query_hypothyroid': [query_hypothyroid],
            'query_hyperthyroid': [query_hyperthyroid],
            'lithium': [lithium],
            'goitre': 0,
            'tumor': [tumor],
            'hypopituitary': [hypopituitary],
            'psych': [psych],
            'TSH': [TSH],
            'TT4': [TT4],
            'T4U': [T4U],
            'FTI': [FTI]
        })

        new_data = self.replace_boolean(new_data)
        new_data_scaled = scaler.transform(new_data[feature_cols])

        prediction = model.predict(new_data_scaled)

        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            print("Result: Probably a healthy thyroid.")
        elif predicted_class == 1:
            print("Result: Suspicion of hypothyroidism.")
        elif predicted_class == 2:
            print("Result: Suspicion of hyperthyroidism.")