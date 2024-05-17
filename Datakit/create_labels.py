import os
import pandas as pd

class Labels:
    def __init__(self,
                base_directory = os.getcwd(),
                train_vs_val_csv = None,
                labels_fname=None):
        self.base_directory = base_directory
        self.labels_fname = labels_fname
        self.path_txt = os.path.join(self.base_directory, 'all_generated' , (self.labels_fname+'.txt'))
        self.path_pbtxt = os.path.join(self.base_directory, 'all_generated' , (self.labels_fname+'.pbtxt'))
        self.df = pd.read_csv(os.path.join(self.base_directory, 'all_generated', train_vs_val_csv))
        self.symptoms = [sym for sym in self.df['index'].values.tolist()]

    def create_pbtxt(self):
        size=[i+1 for i in range(len(self.symptoms))]
        with open(self.path_pbtxt, 'a+') as f:
            for n in size:
                context = "item" + "{\n" + "	id: " + "{}\n".format(n) + "	name: '{}'".format(self.symptoms[n-1]) + "\n}\n"
                print(context)
                f.write(context)
            f.close()
        return

    def create_txt(self):
        with open(self.path_txt, 'a+') as g:
            for symp in self.symptoms:
                context = symp + '\n'
                g.write(context)
            g.close()
        return