import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.utils import shuffle
import click
dest = "/home/pouramini/CO-NNECT/Relation_Classification/data"
valid_dir = "/drive3/pouramini/data/atomic/en_fa/sample_valid_1k"
ulist = ["other", "xAttr", "xIntent", "xNeed", "xWant", "xReact", "xEffect", "oReact", "oEffect", "oWant"]
@click.command()
@click.argument("fname", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--size",
    default=1000,
    type=int,
    help=""
)
@click.option(
    "--out",
    default="",
    type=str,
    help=""
)
@click.option(
    "--tag",
    default="",
    type=str,
    help=""
)
@click.option(
    "--lang",
    default="en",
    type=str,
    help=""
)
def main(path, fname, size, out, tag, lang):
    if fname == "all":
        for lang in ["en", "fa"]:
            for t in ulist[1:]:
                conv(path, "", tag=t, lang=lang, size=size, out=out)
    else:
        conv(path, fname, size, out, tag, lang)

def conv(path, fname, size, out="", tag="", lang="en"):
    other = ""
    if fname:
        tag_fname = path + "/" + fname
    else:
        tag_fname = path + f'/en_fa_{tag}.tsv'
    other_fname = valid_dir + "/en_fa_" + tag + "_other.tsv"
    tag_df = pd.read_table(tag_fname)
    other_df = pd.read_table(other_fname)

    tag_df = pd.concat([tag_df, other_df], ignore_index=True)

    tag_df = tag_df.groupby("prefix").sample(n=size, random_state=1)
    tag_df = tag_df.sample(frac=1, random_state=1)
    tn = "k".join(str(size).rsplit("000", 1))
    tn = tn.replace("kk", "m") 
    #valid = valid.dropna()
    #tag_df = tag_df.dropna()
    print(len(tag_df))
    #labels, uniques = tag_df.prefix.factorize()
    #ulist = uniques.tolist()
    print("labels:", ulist)
    with open(f'{dest}/labels.txt','w') as f:
        for label in ulist:
            print(label.strip(),' ', ulist.index(label), file=f)
    def convert(df, inp):
        with open(f'{inp}.src', 'w') as src,  open(f'{inp}.trg', 'w') as trg:
            _max = 0
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                if lang == "en":
                    s = f'{row["input_text"]} <mask> {row["target_text"]}'
                elif lang == "fa":
                    s = f'{row["input_text_fa"]} <mask> {row["target_text_fa"]}'
                if len(s) > _max:
                    _max = len(s)
                print(s, file= src)
                l = ulist.index(row["prefix"])
                assert l in range(10), "Uknown lable"
                print(l, file= trg)

            print("max length:", _max)
    if not out:
        out = lang + "_" + tag + "_" + tn
    else:
        out = lang + "_" + tag + "_" + tn + "_" + out
    out = dest + "/" + out
    print(out, ":", len(tag_df))
    convert(tag_df, out) 
    print("finished")

if __name__ == "__main__":
    main()

