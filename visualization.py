import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from incremental_repair_utils import *


def uuv_plot_colors(verisig_result_path, dict_color, title='UUV Result'):

    df_verisig = pd.read_csv(verisig_result_path)
    fig, ax = plt.subplots()

    for idx, row in df_verisig.iterrows():
        y_lo, y_hi, h_lo, h_hi = row['y_lo'], row['y_hi'], row['h_lo'], row['h_hi']
        color = dict_color[idx]
        if color == 'red' or color == 'green':
            ax.add_patch(Rectangle((y_lo, h_lo), (y_hi - y_lo), (h_hi - h_lo), facecolor=color, edgecolor=color))
        else:
            ax.add_patch(Rectangle((y_lo, h_lo), (y_hi - y_lo), (h_hi - h_lo), facecolor='none', edgecolor='green', hatch='////', linewidth=0))

    ax.set_xlim([12.0, 22.0])
    ax.set_ylim([10.0, 30.0])
    ax.set_xlabel('init pos y')
    ax.set_ylabel('init heading deg')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return


def mc_plot_colors(verisig_result_path, dict_color, title='MC Result'):

    df_verisig = pd.read_csv(verisig_result_path)
    fig, ax = plt.subplots()

    for idx, row in df_verisig.iterrows():
        pos_lo, pos_hi, vel_lo, vel_hi = row['pos_lo'], row['pos_hi'], row['vel_lo'], row['vel_hi']
        color = dict_color[idx]
        if color == 'red' or color == 'green':
            ax.add_patch(Rectangle((pos_lo, vel_lo), (pos_hi - pos_lo), (vel_hi - vel_lo), facecolor=color, edgecolor=color))
        else:
            ax.add_patch(Rectangle((pos_lo, vel_lo), (pos_hi - pos_lo), (vel_hi - vel_lo), facecolor='none', edgecolor='green', hatch='////', linewidth=0))

    ax.set_xlim([-0.505, 0.395])
    ax.set_ylim([-0.055, 0.045])
    ax.set_xlabel('init position')
    ax.set_ylabel('init velocity')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return


def check_min_robustness(sampled_result_path, dict_color):

    def check_red(region_id):
        return dict_color[region_id] == 'red'

    def check_not_red(region_id):
        return dict_color[region_id] == 'yellow' or dict_color[region_id] == 'green'

    df_sample = pd.read_csv(sampled_result_path)
    red_mask = df_sample['region'].apply(check_red)
    df_red = df_sample[red_mask]
    not_red_mask = df_sample['region'].apply(check_not_red)
    df_not_red= df_sample[not_red_mask]

    # Aggregate by min on robustness, group by region
    df_red_min = df_red.join(df_red.groupby('region')['result'].agg(['min']), on='region')
    print(f'Min robustness of all red regions have mean: {df_red_min["min"].mean()}, std: {df_red_min["min"].std()}')

    df_not_red_min = df_not_red.join(df_not_red.groupby('region')['result'].agg(['min']), on='region')
    print(f'Min robustness of all non-red regions have mean: {df_not_red_min["min"].mean()}, std: {df_not_red_min["min"].std()}')

    df_overall_min = df_sample.join(df_sample.groupby('region')['result'].agg(['min']), on='region')
    print(f'Min robustness overall have mean: {df_overall_min["min"].mean()}, std: {df_overall_min["min"].std()}')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="uuv or mc", default="uuv")
    parser.add_argument("--verisig_result_path", help="path to verisig result csv", default='uuv_verisig_result.csv')
    parser.add_argument("--sampled_result_path", help="path to sampling result csv", default='uuv_sampling_result.csv')
    args = parser.parse_args()

    dict_color = color_regions(args.verisig_result_path, args.sampled_result_path)

    check_min_robustness(args.sampled_result_path, dict_color)

    if args.benchmark == 'uuv':
        uuv_plot_colors(args.verisig_result_path, dict_color)
    elif args.benchmark == 'mc':
        mc_plot_colors(args.verisig_result_path, dict_color)
    else:
        raise NotImplementedError