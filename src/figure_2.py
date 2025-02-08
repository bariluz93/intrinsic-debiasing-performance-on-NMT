import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import glob
sns.set(font="Verdana")
sns.set_style("white")

def main(results_dir):
    files = glob.glob(results_dir+"/*")
    for f in files:
        if f.endswith("de_anti_accuracy.csv"):
            de_anti_accuracy = pd.read_csv(f, index_col=0)
        if f.endswith("he_anti_accuracy.csv"):
            he_anti_accuracy = pd.read_csv(f, index_col=0)
        if f.endswith("ru_anti_accuracy.csv"):
            ru_anti_accuracy = pd.read_csv(f, index_col=0)
        if f.endswith("de_bleu.csv"):
            de_bleu = pd.read_csv(f, index_col=0)
        if f.endswith("he_bleu.csv"):
            he_bleu = pd.read_csv(f, index_col=0)
        if f.endswith("ru_bleu.csv"):
            ru_bleu = pd.read_csv(f, index_col=0)
    df = pd.DataFrame([[float("{:.2f}".format(de_bleu['Hard Debias']['Encoder Input']-de_bleu['Orig']['Encoder Input'])), float("{:.2f}".format(de_anti_accuracy['Hard Debias']['Encoder Input']-de_anti_accuracy['Orig']['Encoder Input'])), 'de-encoder-hard'],
                       [float("{:.2f}".format(he_bleu['Hard Debias']['Encoder Input']-he_bleu['Orig']['Encoder Input'])), float("{:.2f}".format(he_anti_accuracy['Hard Debias']['Encoder Input']-he_anti_accuracy['Orig']['Encoder Input'])), 'he-encoder-hard'],
                       [float("{:.2f}".format(ru_bleu['Hard Debias']['Encoder Input']-ru_bleu['Orig']['Encoder Input'])), float("{:.2f}".format(ru_anti_accuracy['Hard Debias']['Encoder Input']-ru_anti_accuracy['Orig']['Encoder Input'])), 'ru-encoder-hard'],

                       [0, 0, ''],

                       [float("{:.2f}".format(de_bleu['Hard Debias']['Decoder Input']-de_bleu['Orig']['Decoder Input'])), float("{:.2f}".format(de_anti_accuracy['Hard Debias']['Decoder Input']-de_anti_accuracy['Orig']['Decoder Input'])), 'de-decoder-in-hard'],
                       [float("{:.2f}".format(he_bleu['Hard Debias']['Decoder Input']-he_bleu['Orig']['Decoder Input'])), float("{:.2f}".format(he_anti_accuracy['Hard Debias']['Decoder Input']-he_anti_accuracy['Orig']['Decoder Input'])), 'he-decoder-in-hard'],

                       [0, 0, ' '],

                       [float("{:.2f}".format(de_bleu['Hard Debias']['Decoder Output']-de_bleu['Orig']['Decoder Output'])), float("{:.2f}".format(de_anti_accuracy['Hard Debias']['Decoder Output']-de_anti_accuracy['Orig']['Decoder Output'])), 'de-decoder-out-hard'],
                       [float("{:.2f}".format(he_bleu['Hard Debias']['Decoder Output']-he_bleu['Orig']['Decoder Output'])), float("{:.2f}".format(he_anti_accuracy['Hard Debias']['Decoder Output']-he_anti_accuracy['Orig']['Decoder Output'])), 'he-decoder-out-hard'],

                       [0, 0, '  '],
                       [0, 0, '   '],

                       [float("{:.2f}".format(de_bleu['INLP']['Encoder Input']-de_bleu['Orig']['Encoder Input'])), float("{:.2f}".format(de_anti_accuracy['INLP']['Encoder Input']-de_anti_accuracy['Orig']['Encoder Input'])), 'de-encoder-inlp'],
                       [float("{:.2f}".format(he_bleu['INLP']['Encoder Input']-he_bleu['Orig']['Encoder Input'])), float("{:.2f}".format(he_anti_accuracy['INLP']['Encoder Input']-he_anti_accuracy['Orig']['Encoder Input'])), 'he-encoder-inlp'],
                       [float("{:.2f}".format(ru_bleu['INLP']['Encoder Input']-ru_bleu['Orig']['Encoder Input'])), float("{:.2f}".format(ru_anti_accuracy['INLP']['Encoder Input']-ru_anti_accuracy['Orig']['Encoder Input'])), 'ru-encoder-inlp'],

                       [0, 0, '    '],

                       [float("{:.2f}".format(de_bleu['INLP']['Decoder Input']-de_bleu['Orig']['Decoder Input'])), float("{:.2f}".format(de_anti_accuracy['INLP']['Decoder Input']-de_anti_accuracy['Orig']['Decoder Input'])), 'de-decoder-in-inlp'],
                       [float("{:.2f}".format(he_bleu['INLP']['Decoder Input']-he_bleu['Orig']['Decoder Input'])), float("{:.2f}".format(he_anti_accuracy['INLP']['Decoder Input']-he_anti_accuracy['Orig']['Decoder Input'])), 'he-decoder-in-inlp'],

                       [0, 0, '     '],

                       [float("{:.2f}".format(de_bleu['INLP']['Decoder Output']-de_bleu['Orig']['Decoder Output'])), float("{:.2f}".format(de_anti_accuracy['INLP']['Decoder Output']-de_anti_accuracy['Orig']['Decoder Output'])), 'de-decoder-out-inlp'],
                       [float("{:.2f}".format(he_bleu['INLP']['Decoder Output']-he_bleu['Orig']['Decoder Output'])), float("{:.2f}".format(he_anti_accuracy['INLP']['Decoder Output']-he_anti_accuracy['Orig']['Decoder Output'])), 'he-decoder-out-inlp']],

                      columns=['delta BLEU', 'delta acc', 'lang'])

    ax = sns.barplot(x=df["lang"], y=df["delta acc"], color=sns.color_palette()[1])
    ax = sns.barplot(x=df["lang"], y=df["delta BLEU"], color=sns.color_palette()[0])

    # plt.plot([0, 0], [0, 5], linewidth=2)
    ax.axhline(0, color='black')
    ax.set(xlabel='', ylabel='relative change')

    # plt.xticks(rotation=90)

    labels = ['$\Delta$ acc', '$\Delta$ BLEU']
    # plt.legend(loc='upper left', labels=['$\Delta$ acc', '$\Delta$ BLEU'], facecolor='w')
    handles = []
    for color, label in zip([sns.color_palette()[1], sns.color_palette()[0]], labels):
     handles.append(plt.scatter([], [], c=color, label=label))
    plt.legend(handles=handles)

    for ind, bars in enumerate(ax.patches):
     if ind < 20:
      c = color = sns.color_palette()[1]
      ext = 0.08
     else:
      c = color = sns.color_palette()[0]
      ext = -0.4

     skip = [3, 6, 9, 10, 14, 17, 23, 26, 29, 30, 34, 37]
     if ind not in skip:
      if ind in [4, 5, 7, 8]:
       ax.text(bars.get_x(), bars.get_height() + 0.25, bars.get_height(), fontsize=7, color=c)
      elif ind in [11, 12, 13]:
       ax.text(bars.get_x(), bars.get_height() - 0.5, bars.get_height(), fontsize=7, color=c)
      elif ind in [31, 32, 33]:
       ax.text(bars.get_x(), bars.get_height() + 0.3, bars.get_height(), fontsize=7, color=c)
      else:
       ax.text(bars.get_x() + 0.15, bars.get_height() + ext, bars.get_height(), fontsize=7, color=c)

    plt.xticks([0, 1, 2, 4, 5, 7, 8, 11, 12, 13, 15, 16, 18, 19])

    ax.set_xticklabels(['de', 'he', 'ru', 'de', 'he', 'de', 'he', 'de', 'he', 'ru', 'de', 'he', 'de', 'he'])

    ax.text(0.5, -11, 'Enc', fontsize=9)
    ax.text(3.5, -11, 'Dec_in', fontsize=9)
    ax.text(6.5, -11, 'Dec_out', fontsize=9)

    ax.text(11.5, -11, 'Enc', fontsize=9)
    ax.text(14.5, -11, 'Dec_in', fontsize=9)
    ax.text(17.5, -11, 'Dec_out', fontsize=9)

    ax.text(1.7, 4.5, 'Hard Debiasing', fontsize=12)
    ax.text(14.5, 4.5, 'INLP', fontsize=12)

    ax.axvline(9.5, ls='--', c='black')

    plt.tight_layout()

    plt.savefig("acc-bleu.pdf")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--results_dir', type=str, required=True,
        help="path to the results dir")
    args = parser.parse_args()
    main(args.results_dir)