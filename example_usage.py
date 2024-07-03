# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:54:02 2024

@author: BHR
"""

import os
import pandas as pd
import numpy as np
from kmodes.kprototypes_sampling_mechanism import kprototype_sampled_analysis


class DataFrameGenerator:
    def __init__(self, file_path, n_rows=30000, n_cols=20):
        self.file_path = file_path
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.df = None

    def check_and_load_data(self):
        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path)
        else:
            self.df = self.generate_random_dataframe()
            self.save_dataframe()

    def generate_random_dataframe(self):
        np.random.seed(42)  # For reproducibility
        data = {}

        # Generate categorical data
        data['Gender'] = np.random.choice(['M', 'F'], self.n_rows)
        data['Education'] = np.random.choice(['Elementary', 'High School', 'BSc', 'MSc', 'PhD'], self.n_rows)
        careers = [
            'Driver', 'Teacher', 'Professor', 'Engineer', 'Doctor', 'Nurse', 'Lawyer', 'Artist', 'Scientist',
            'Chef', 'Mechanic', 'Pilot', 'Pharmacist', 'Journalist', 'Photographer', 'Actor', 'Musician', 'Dancer',
            'Writer', 'Architect', 'Electrician'
        ]
        data['Career'] = np.random.choice(careers, self.n_rows)
        data['Income'] = np.random.choice(['Low', 'Mid', 'High'], self.n_rows)

        # Generate numerical data for the remaining columns
        for i in range(4, self.n_cols):
            data[f'Numerical_{i-3}'] = np.random.uniform(0, 100, self.n_rows)

        return pd.DataFrame(data)

    def save_dataframe(self):
        self.df.to_csv(self.file_path, index=False)

    def get_dataframe(self):
        return self.df




def main():
    file_path = 'sample_data.csv'
    generator = DataFrameGenerator(file_path)
    generator.check_and_load_data()
    df = generator.get_dataframe()
    df=df.iloc[:1000,:]
    kpsa=kprototype_sampled_analysis()
    res=kpsa.fit_predict(df.values,categorical=[0,1,2,3])
    temp=1


if __name__ == "__main__":
    main()


