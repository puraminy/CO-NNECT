import pandas as pd
from tqdm import tqdm
from pathlib import Path

data_dir = "/drive2/pouramini/pet/data/atomic"
def conv(tag):
    train_fname = data_dir + f'/atomic_train_nn_5k_per_prefix_{tag}_other.tsv'
    valid_fname = data_dir + f'/atomic_validation_nn_1k_per_prefix_{tag}_other.tsv'
    train = pd.read_table(train_fname)
    valid = pd.read_table(valid_fname)
    #valid = valid.dropna()
    #train = train.dropna()
    print(len(train))
    print(len(valid))
    labels, uniques = train.prefix.factorize()
    ulist = uniques.tolist()
    print("labels:", ulist)
    with open(f'labels_{tag}.txt','w') as f:
        for label in ulist:
            print(label.strip(),' ', ulist.index(label), file=f)
    def convert(df, inp):
        with open(f'{inp}.src', 'w') as src,  open(f'{inp}.trg', 'w') as trg:
            _max = 0
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                s = f'{row["input_text"]} <mask> {row["target_text"]}'
                if len(s) > _max:
                    _max = len(s)
                print(s, file= src)
                l = ulist.index(row["prefix"])
                assert l in range(9), "Uknown lable"
                print(l, file= trg)

            print("max length:", _max)

    convert(train, Path(train_fname).stem)
    convert(valid, Path(valid_fname).stem)
    print("finished")


l = ["xIntent", "xNeed", "xWant", "xReact", "xEffect", "oReact", "oEffect", "oWant"]

for t in l:
    conv(t)
