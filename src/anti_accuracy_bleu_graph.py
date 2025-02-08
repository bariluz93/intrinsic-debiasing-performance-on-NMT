import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from consts import LANGUAGE_STR_MAP, OUTPUTS_HOME
import csv
# plt.style.use('seaborn-dark-palette')
LANGUAGES = LANGUAGE_STR_MAP.values()
LANGUAGES = list(LANGUAGES)
LANGUAGES.remove('es')

plt.rcParams['font.family'] = 'serif'
#plt.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = False
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def crate_anti_accuracy_per_bleu_graph(results_dir):
    # x=[1,2,3,4,5,6]
    # y=[1,2,3,4,5,6]
    #
    # plt.plot(x[0], y[0], marker='o',label="marker='{0}'".format('o'),markerfacecolor='r',markeredgecolor='r')
    # plt.plot(x[1], y[1], marker='o',label="marker='{0}'".format('o'),markerfacecolor='none', markeredgecolor='r')
    # plt.plot(x[2], y[2], marker='s',label="marker='{0}'".format('s'),markerfacecolor='b',markeredgecolor='b')
    # plt.plot(x[3], y[3], marker='s',label="marker='{0}'".format('s'),markerfacecolor='none', markeredgecolor='b')
    # plt.plot(x[4], y[4], marker='^',label="marker='{0}'".format('^'),markerfacecolor='g', markeredgecolor='g')
    # plt.plot(x[5], y[5], marker='^',label="marker='{0}'".format('^'),markerfacecolor='none', markeredgecolor='g')
    # plt.show()
    x_encoder_hard_debias_list, y_encoder_hard_debias_list = [],[]
    x_encoder_inlp_list, y_encoder_inlp_list = [],[]
    x_decoder_hard_debias_list, y_decoder_hard_debias_list = [],[]
    x_decoder_inlp_list, y_decoder_inlp_list = [],[]
    for language in LANGUAGES:
        with open(results_dir+"results_EASY_NMT_"+language+"_anti_accuracy.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            data = [row for row in spamreader]

            encoder_input_res,decoder_input_res,decoder_output_res=data[1],data[2],data[3]
            orig_encoder_input_res,hard_debiased_encoder_input_res,inlp_encoder_input_res = float(encoder_input_res[1]),float(encoder_input_res[2]),float(encoder_input_res[3])
            orig_decoder_input_res,hard_debiased_decoder_input_res,inlp_decoder_input_res = float(decoder_input_res[1]),float(decoder_input_res[2]),float(decoder_input_res[3])
            orig_decoder_output_res,hard_debiased_decoder_output_res,inlp_decoder_output_res = float(decoder_output_res[1]),float(decoder_output_res[2]),float(decoder_output_res[3])

            x_hard_debiased_encoder_input = hard_debiased_encoder_input_res - orig_encoder_input_res
            x_inlp_encoder_input = inlp_encoder_input_res-orig_encoder_input_res
            if language != 'ru' and language != 'es':
                x_hard_debiased_decoder_input = hard_debiased_decoder_input_res-orig_decoder_input_res
                x_inlp_decoder_input = inlp_decoder_input_res-orig_decoder_input_res

                x_hard_debiased_decoder_output = hard_debiased_decoder_output_res-orig_decoder_output_res
                x_inlp_decoder_output = inlp_decoder_output_res-orig_decoder_output_res

                x_hard_debiased_decoder = (x_hard_debiased_decoder_input+x_hard_debiased_decoder_output)/2
                x_inlp_decoder = (x_inlp_decoder_input+ x_inlp_decoder_output)/2

        with open(results_dir+"results_EASY_NMT_"+language+"_bleu.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            data = [row for row in spamreader]
            encoder_input_bleu,decoder_input_bleu,decoder_output_bleu=data[1],data[2],data[3]
            orig_encoder_input_bleu,hard_debiased_encoder_input_bleu,inlp_encoder_input_bleu = float(encoder_input_bleu[1]),float(encoder_input_bleu[2]),float(encoder_input_bleu[3])
            if language != 'ru' and language != 'es':
                orig_decoder_input_bleu,hard_debiased_decoder_input_bleu,inlp_decoder_input_bleu = float(decoder_input_bleu[1]),float(decoder_input_bleu[2]),float(decoder_input_bleu[3])
                orig_decoder_output_bleu,hard_debiased_decoder_output_bleu,inlp_decoder_output_bleu = float(decoder_output_bleu[1]),float(decoder_output_bleu[2]),float(decoder_output_bleu[3])

            y_hard_debiased_encoder_input = hard_debiased_encoder_input_bleu - orig_encoder_input_bleu
            y_inlp_encoder_input = inlp_encoder_input_bleu - orig_encoder_input_bleu

            if language != 'ru' and language != 'es':
                y_hard_debiased_decoder_input = hard_debiased_decoder_input_bleu - orig_decoder_input_bleu
                y_inlp_decoder_input = inlp_decoder_input_bleu -orig_decoder_input_bleu

                y_hard_debiased_decoder_output = hard_debiased_decoder_output_bleu - orig_decoder_output_bleu
                y_inlp_decoder_output = inlp_decoder_output_bleu - orig_decoder_output_bleu

                y_hard_debiased_decoder = (y_hard_debiased_decoder_input+y_hard_debiased_decoder_output)/2
                y_inlp_decoder = (y_inlp_decoder_input+y_inlp_decoder_output)/2

        x_encoder_hard_debias_list.append(x_hard_debiased_encoder_input)
        y_encoder_hard_debias_list.append(y_hard_debiased_encoder_input)
        x_encoder_inlp_list.append(x_inlp_encoder_input)
        y_encoder_inlp_list.append(y_inlp_encoder_input)

        # plt.plot(x_hard_debiased_encoder_input, y_hard_debiased_encoder_input, marker='.', markerfacecolor='g', markeredgecolor='g', markersize=18,alpha=.5)
        # plt.plot(x_inlp_encoder_input, y_inlp_encoder_input, marker='*', markerfacecolor='g', markeredgecolor='g', markersize=18,alpha=.5)
        if language!='ru' and language!='es':
            # plt.plot(x_hard_debiased_decoder, y_hard_debiased_decoder, marker='.', markerfacecolor='b', markeredgecolor='b', markersize=18,alpha=.5)
            # plt.plot(x_inlp_decoder, y_inlp_decoder, marker='*', markerfacecolor='b', markeredgecolor='b', markersize=18,alpha=.5)
            x_decoder_hard_debias_list.append(x_hard_debiased_decoder)
            y_decoder_hard_debias_list.append(y_hard_debiased_decoder)
            x_decoder_inlp_list.append(x_inlp_decoder)
            y_decoder_inlp_list.append(y_inlp_decoder)



    plt.scatter(x_encoder_hard_debias_list,y_encoder_hard_debias_list,c="g",marker='o', s=100,label='Encoder, Hard Debias')
    plt.scatter(x_encoder_inlp_list, y_encoder_inlp_list, c="g", marker='*',s=100, label='Encoder, INLP')
    # plot encoder line
    x_plot_encoder =x_encoder_hard_debias_list+x_encoder_inlp_list
    y_plot_encoder = y_encoder_hard_debias_list+y_encoder_inlp_list
    plt.plot(sorted(x_plot_encoder),[y for _,y in sorted(zip(x_plot_encoder,y_plot_encoder))],c="g")

    plt.scatter(x_decoder_hard_debias_list, y_decoder_hard_debias_list, c="b", marker='o',s=100, label='Decoder, Hard Debias')
    plt.scatter(x_decoder_inlp_list, y_decoder_inlp_list, c="b", marker='*',s=100, label='Decoder , INLP')
    # plot decoder line
    x_plot_decoder = x_decoder_hard_debias_list + x_decoder_inlp_list
    y_plot_decoder = y_decoder_hard_debias_list + y_decoder_inlp_list
    plt.plot(sorted(x_plot_decoder), [y for _, y in sorted(zip(x_plot_decoder, y_plot_decoder))], c="b")

    # plot hard debias line
    x_plot_hard_debias = x_encoder_hard_debias_list + x_decoder_hard_debias_list
    y_plot_hard_debias = y_encoder_hard_debias_list + y_decoder_hard_debias_list
    plt.plot(sorted(x_plot_hard_debias), [y for _, y in sorted(zip(x_plot_hard_debias, y_plot_hard_debias))], c="darkorchid",linewidth=20, alpha=0.2)

    # plot inlp line
    x_plot_inlp = x_encoder_inlp_list + x_decoder_inlp_list
    y_plot_inlp = y_encoder_inlp_list + y_decoder_inlp_list
    plt.plot(sorted(x_plot_inlp), [y for _, y in sorted(zip(x_plot_inlp, y_plot_inlp))], c="orange",linewidth=20, alpha=0.2)

    plt.xlabel("Delta Anti Accuracy")
    plt.ylabel("Delta Bleu")
    # encoder_input_hard_debias_german = mlines.Line2D([], [], color=colors[0], marker='*', linestyle='None',
    #                           markersize=10, label='Blue stars')
    #
    #
    # plt.legend(handles=[blue_star, red_square, purple_triangle])
    # legend_elements = [Line2D([0], [0], marker='o', color='g', label='Encoder, Hard Debias', markersize=15),
    #                    Line2D([0], [0], marker='o', color='b', label='Decoder, Hard Debias', markersize=15),
    #                    Line2D([0], [0], marker='*', color='g', label='Encoder, INLP', markersize=15),
    #                    Line2D([0], [0], marker='*', color='b', label='Decoder, INLP', markersize=15)]

    # plt.legend(handles=legend_elements)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    crate_anti_accuracy_per_bleu_graph(OUTPUTS_HOME+"results/11-10-2022_15-08-28_1_token_professions_by_pca_false/")