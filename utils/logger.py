'''
simple logging class
writes to csv
'''

import os
import csv
from datetime import datetime

class Logger:
    def __init__(self,parent_dir,colnames) -> None:
        self.log_dir = os.path.join(parent_dir,'log')
        self.log_filename = f"{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        if os.name == 'nt':  # 'nt' indicates Windows
            # Replace colons with hyphens for Windows
            self.log_filename = self.log_filename.replace(":", "-")
        self.log_path = os.path.join(self.log_dir, self.log_filename)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self._initialize_csv(colnames)

    def _initialize_csv(self,colnames):
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(colnames)

    def log(self,*args):
        with open(self.log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(args)