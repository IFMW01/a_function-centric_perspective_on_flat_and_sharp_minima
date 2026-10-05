import os
import json
import numpy as np
import pandas as pd
import math
import scipy
import pingouin as pg
from functools import partial
from scipy.special import softmax as softmax_fn
from scipy.special import kl_div


def simple_mean_sem(dist):
    np_dist = np.asarray(dist)
    mean = np.mean(np_dist)
    sem = np.std(np_dist) / math.sqrt(np_dist.size)
    return mean, sem


def combine_columns(column, row):
    mean_column = f"{column} Mean"
    sem_column = f"{column} SEM"
    if column == "SAM Sharpness":
        return f"{row[mean_column]:.3E} ({row[sem_column]:.3E})"
    else:
        return f"{row[mean_column]:.3f} ({row[sem_column]:.3f})"


def load_json(path):
    with open(path) as f:
        dict = json.load(f)
    return dict


def activation_dist(model1, model2, softmax=False):
    model1 = model1.astype(float)
    model2 = model2.astype(float)
    if softmax:
        model1 = softmax_fn(model1, axis=1)
        model2 = softmax_fn(model2, axis=1)

    distances = np.sqrt(np.sum(np.square(model1 - model2), axis=1))
    return distances.mean()


def pred_disagreement(base_preds, logits_preds):
    return 1 - (
        np.sum(np.equal(np.argmax(base_preds, axis=1), np.argmax(logits_preds, axis=1)))
        / len(base_preds)
    )


def JSD(model_1_logits, model_2_logits, multiplier=10, softmax=False):
    if softmax == True:
        model_1_softmax = model_1_logits
        model_2_softmax = model_2_logits
    else:
        model_1_softmax = softmax_fn(model_1_logits.astype(np.float64), axis=1)
        model_2_softmax = softmax_fn(model_2_logits.astype(np.float64), axis=1)
    m = (model_1_softmax + model_2_softmax) / 2.0
    left = kl_div(model_1_softmax, m)
    right = kl_div(model_2_softmax, m)
    left_sum = np.sum(left, axis=1, keepdims=True)
    right_sum = np.sum(right, axis=1, keepdims=True)
    js = 0.5 * (left_sum + right_sum)
    return js.mean()


def sharp_metric_significance(pth, extension, epoch):
    gen_gap = []
    ece = []
    pred_dis = []
    acc = []
    corrupt = []
    fr_norm = []
    wn = []
    ss = []
    rf = []
    for i in range(0, 10):
        path = pth + f"{i}/" + extension + ".json"
        sharpness_path = pth + f"{i}/" + extension + "_sharpness.json"
        corruption_path = pth + f"{i}/" + extension + "_corruption.json"
        print(path)
        with open(path) as f:
            model_results = json.load(f)
        with open(corruption_path) as f:
            corruption_results = json.load(f)
        with open(sharpness_path) as f:
            sharpness_results = json.load(f)
        fr_norm.append(sharpness_results["model_eot"]["fisher_rao_norm"])
        wn.append(sharpness_results["model_eot"]["squared_euclidean_norm"])
        ss.append(sharpness_results["model_eot"]["sam_sharpness"])
        rf.append(sharpness_results["model_eot"]["relative_flatness"])

        ece.append(sharpness_results["model_eot"]["ece"])

        acc.append(sharpness_results["model_eot"]["acc"])
        gen_gap.append(
            model_results[f"{epoch}"]["train accuracy"]
            - (100 * sharpness_results["model_eot"]["acc"])
        )
        counter = 0
        total_acc = 0
        for key in corruption_results:
            for inner_key in corruption_results[key]:
                counter += 1
                total_acc += corruption_results[key][inner_key]["accuracy"]
        corrupt.append(total_acc / counter)
        for j in range(0, 10):
            if i == j:
                continue
            else:
                compare_path = pth + f"{j}/" + extension + ".json"
                with open(compare_path) as f:
                    compare_model_results = json.load(f)
                pred_dis.append(
                    pred_disagreement(
                        np.array(model_results[f"{epoch}"]["function"]),
                        np.array(compare_model_results[f"{epoch}"]["function"]),
                    )
                )

    output_dict = {}
    output_dict["Generalisation Gap"] = gen_gap
    output_dict["Test Accuracy"] = acc
    output_dict["Test ECE"] = ece
    output_dict["Corruption Accuracy"] = corrupt
    output_dict["Prediction Disagreement"] = pred_dis
    output_dict["SAM Sharpness"] = ss
    output_dict["Fisher-Rao norm"] = fr_norm
    output_dict["Relative Flatness"] = rf

    return output_dict


def sharp_imagenet_metric_significance(pth, extension, epoch):
    output_dict = {}
    ece = []
    gen_gap = []
    corrupt = []
    ad_dist = []
    js_dist = []
    pred_dis = []
    acc = []
    fr_norm = []
    wn = []
    ss = []
    rf = []
    for i in range(0, 10):
        path = pth + f"{i}/" + extension + ".json"
        sharpness_path = pth + f"{i}/" + extension + "_sharpness.json"
        print(path)
        corruption_path = pth + f"{i}/" + extension + "_corruption.json"
        with open(corruption_path) as f:
            corruption_results = json.load(f)
        with open(path) as f:
            model_results = json.load(f)
        with open(sharpness_path) as f:
            sharpness_results = json.load(f)
        fr_norm.append(sharpness_results["model_eot"]["fisher_rao_norm"])
        wn.append(sharpness_results["model_eot"]["squared_euclidean_norm"])
        ss.append(sharpness_results["model_eot"]["sam_sharpness"])
        gen_gap.append(
            model_results[f"{epoch}"]["train accuracy"]
            - (100 * sharpness_results["model_eot"]["acc"])
        )

        ece.append(sharpness_results["model_eot"]["ece"])

        acc.append(sharpness_results["model_eot"]["acc"])
        counter = 0
        total_acc = 0
        for key in corruption_results:
            for inner_key in corruption_results[key]:
                counter += 1
                total_acc += corruption_results[key][inner_key]["accuracy"]
        corrupt.append(total_acc / counter)

        for j in range(0, 10):
            if i == j:
                continue
            else:
                compare_path = pth + f"{j}/" + extension + ".json"
                with open(compare_path) as f:
                    compare_model_results = json.load(f)
                pred_dis.append(
                    pred_disagreement(
                        np.array(model_results[f"{epoch}"]["function"]),
                        np.array(compare_model_results[f"{epoch}"]["function"]),
                    )
                )
                js_dist.append(
                    (
                        JSD(
                            np.array(model_results[f"{epoch}"]["function"]),
                            np.array(compare_model_results[f"{epoch}"]["function"]),
                        )
                    )
                )
                ad_dist.append(
                    (
                        activation_dist(
                            np.array(model_results[f"{epoch}"]["function"]),
                            np.array(compare_model_results[f"{epoch}"]["function"]),
                        )
                    )
                )

    output_dict["Generalisation Gap"] = gen_gap
    output_dict["Test Accuracy"] = acc
    output_dict["Test ECE"] = ece
    output_dict["Corruption Accuracy"] = corrupt
    output_dict["Prediction Disagreement"] = pred_dis
    output_dict["SAM Sharpness"] = ss
    output_dict["Fisher-Rao norm"] = fr_norm

    gen_gap_mean, gen_gap_sem = simple_mean_sem(gen_gap)
    acc_mean, acc_sem = simple_mean_sem(acc)
    ece_mean, ece_sem = simple_mean_sem(ece)
    corrupt_mean, corrupt_sem = simple_mean_sem(corrupt)
    pred_dis_mean, pred_dis_sem = simple_mean_sem(pred_dis)
    js_dist_mean, js_dist_sem = simple_mean_sem(js_dist)
    ad_dist_mean, ad_dist_sem = simple_mean_sem(ad_dist)
    fr_dist_mean, fr_dist_sem = simple_mean_sem(fr_norm)
    ss_dist_mean, ss_dist_sem = simple_mean_sem(ss)

    arr = np.array(
        [
            gen_gap_mean,
            gen_gap_sem,
            acc_mean,
            acc_sem,
            ece_mean,
            ece_sem,
            corrupt_mean,
            corrupt_sem,
            pred_dis_mean,
            pred_dis_sem,
            js_dist_mean,
            js_dist_sem,
            ad_dist_mean,
            ad_dist_sem,
            fr_dist_mean,
            fr_dist_sem,
            ss_dist_mean,
            ss_dist_sem,
        ]
    )
    df = pd.DataFrame([arr])
    df.columns = [
        "Generalisation Gap Mean",
        "Generalisation Gap SEM",
        "Test Accuracy Mean",
        "Test Accuracy SEM",
        "Test ECE Mean",
        "Test ECE SEM",
        "Corruption Accuracy Mean",
        "Corruption Accuracy SEM",
        "Prediction Disagreement Mean",
        "Prediction Disagreement SEM",
        "JS Divergence Mean",
        "JS Divergence SEM",
        "Activation Distance Mean",
        "Activation Distance SEM",
        "Fisher Rao Norm Mean",
        "Fisher Rao Norm SEM",
        "SAM Sharpness Mean",
        "SAM Sharpness SEM",
    ]
    return output_dict, df


def create_processed_table_cifar(dataset, architecture, epochs=99):
    base_df = sharp_metric_significance(
        f"./models/{dataset}/{architecture}_base/", f"{architecture}_base", epochs
    )
    base_SAM_df = sharp_metric_significance(
        f"./models/{dataset}/{architecture}_base_SAM/",
        f"{architecture}_base_SAM",
        epochs,
    )
    aug_df = sharp_metric_significance(
        f"./models/{dataset}/{architecture}_aug/", f"{architecture}_aug", epochs
    )
    aug_SAM_df = sharp_metric_significance(
        f"./models/{dataset}/{architecture}_aug_SAM/", f"{architecture}_aug_SAM", epochs
    )
    wd_df = sharp_metric_significance(
        f"./models/{dataset}/{architecture}_wd/", f"{architecture}_wd", epochs
    )
    wd_SAM_df = sharp_metric_significance(
        f"./models/{dataset}/{architecture}_wd_SAM/", f"{architecture}_wd_SAM", epochs
    )

    combined_df = pd.concat(
        [base_df, base_SAM_df, aug_df, aug_SAM_df, wd_df, wd_SAM_df], ignore_index=True
    )
    combined_df.to_csv(f"./models/{dataset}/{architecture}.csv")
    columns = [
        "Generalisation Gap",
        "Test Accuracy",
        "Test ECE",
        "Corruption Accuracy",
        "Prediction Disagreement",
        "Fisher Rao Norm",
        "SAM Sharpness",
        "Relative Flatness",
    ]
    presenting_table = pd.DataFrame()

    for column in columns:
        presenting_table[column] = combined_df.apply(
            partial(combine_columns, column), axis=1
        )

    presenting_table.to_csv(f"./models/{dataset}/{architecture}_processed.csv")

    return base_df, base_SAM_df, aug_df, aug_SAM_df, wd_df, wd_SAM_df


def create_processed_table_SAM(architecture, epochs=99):
    base_df = sharp_metric_significance(
        f"./models/SAM_rho/{architecture}_Base/", f"{architecture}_Base", epochs
    )
    base_SAM_5 = sharp_metric_significance(
        f"./models/SAM_rho/{architecture}_Base_SAM_0.5/",
        f"{architecture}_Base_SAM_0.5",
        epochs,
    )
    base_SAM_25 = sharp_metric_significance(
        f"./models/SAM_rho/{architecture}_Base_SAM_0.25/",
        f"{architecture}_Base_SAM_0.25",
        epochs,
    )
    base_SAM_05 = sharp_metric_significance(
        f"./models/SAM_rho/{architecture}_Base_SAM_0.05/",
        f"{architecture}_Base_SAM",
        epochs,
    )
    base_SAM_025 = sharp_metric_significance(
        f"./models/SAM_rho/{architecture}_Base_SAM_0.025/",
        f"{architecture}_Base_SAM_0.025",
        epochs,
    )
    base_SAM_005 = sharp_metric_significance(
        f"./models/SAM_rho/{architecture}_Base_SAM_0.005/",
        f"{architecture}_Base_SAM",
        epochs,
    )
    base_SAM_0025 = sharp_metric_significance(
        f"./models/SAM_rho/{architecture}_Base_SAM_0.0025/",
        f"{architecture}_Base_SAM_0.0025",
        epochs,
    )

    combined_df = pd.concat(
        [
            base_df,
            base_SAM_5,
            base_SAM_25,
            base_SAM_05,
            base_SAM_025,
            base_SAM_005,
            base_SAM_0025,
        ],
        ignore_index=True,
    )
    combined_df.to_csv(f"./SAM_rho_{architecture}.csv")
    columns = [
        "Generalisation Gap",
        "Test Accuracy",
        "Test ECE",
        "Corruption Accuracy",
        "Prediction Disagreement",
        "Fisher Rao Norm",
        "SAM Sharpness",
        "Relative Flatness",
    ]
    presenting_table = pd.DataFrame()

    for column in columns:
        presenting_table[column] = combined_df.apply(
            partial(combine_columns, column), axis=1
        )
    presenting_table
    presenting_table.to_csv(f"./SAM_rho_{architecture}_processed.csv")

    return (
        base_df,
        base_SAM_5,
        base_SAM_25,
        base_SAM_05,
        base_SAM_025,
        base_SAM_005,
        base_SAM_0025,
    )


def create_processed_table_imagenet(dataset, architecture, epochs=99):
    base_dict, base_df = sharp_imagenet_metric_significance(
        f"./models/{dataset}/{architecture}_Base/", f"{architecture}_Base", epochs
    )
    base_SAM_dict, base_SAM_df = sharp_imagenet_metric_significance(
        f"./models/{dataset}/{architecture}_Base_SAM/",
        f"{architecture}_Base_SAM",
        epochs,
    )
    aug_dict, aug_df = sharp_imagenet_metric_significance(
        f"./models/{dataset}/{architecture}_aug/", f"{architecture}_aug", epochs
    )
    aug_SAM_dict, aug_SAM_df = sharp_imagenet_metric_significance(
        f"./models/{dataset}/{architecture}_aug_SAM/", f"{architecture}_aug_SAM", epochs
    )
    wd_dict, wd_df = sharp_imagenet_metric_significance(
        f"./models/{dataset}/{architecture}_wd/", f"{architecture}_wd", epochs
    )
    wd_SAM_dict, wd_SAM_df = sharp_imagenet_metric_significance(
        f"./models/{dataset}/{architecture}_wd_SAM/", f"{architecture}_wd_SAM", epochs
    )

    combined_df = pd.concat(
        [base_df, base_SAM_df, aug_df, aug_SAM_df, wd_df, wd_SAM_df], ignore_index=True
    )
    combined_df.to_csv(f"./{dataset}_{architecture}.csv")
    columns = [
        "Generalisation Gap",
        "Test Accuracy",
        "Test ECE",
        "Corruption Accuracy",
        "Prediction Disagreement",
        "Fisher Rao Norm",
        "SAM Sharpness",
    ]
    presenting_table = pd.DataFrame()

    for column in columns:
        presenting_table[column] = combined_df.apply(
            partial(combine_columns, column), axis=1
        )
    presenting_table
    presenting_table.to_csv(f"./{dataset}_{architecture}_processed.csv")
    return base_dict, base_SAM_dict, aug_dict, aug_SAM_dict, wd_dict, wd_SAM_dict


def hypothesis_test(baseline, compare, pvalue_threshold=0.05, alternative="less"):
    if alternative == "less":
        result = pg.wilcoxon(compare, baseline, alternative=alternative)
    if alternative == "greater":
        result = pg.wilcoxon(compare, baseline, alternative=alternative)
    return [result["p-val"].iloc[0], result["RBC"].iloc[0]]


def hypothesis_tables_cifar(baseline_dict, comapre_dict, alpha=0.05):
    table_data = {}
    saftey_data = {}
    sharp_data = {}
    saftey_results = {}
    sharp_results = {}
    for key in baseline_dict.keys():
        if key == "Generalisation Gap":
            test_results = hypothesis_test(
                baseline_dict[key], comapre_dict[key], alternative="less"
            )
            if test_results[0] < alpha:
                table_data[key] = f"\\cmark ({test_results[1]})"
            else:
                table_data[key] = f"\\xmark (N/A)"
        elif key == "Test Accuracy":
            test_results = hypothesis_test(
                baseline_dict[key], comapre_dict[key], alternative="greater"
            )
            if test_results[0] < alpha:
                table_data[key] = f"\\cmark ({test_results[1]})"
            else:
                table_data[key] = f"\\xmark (N/A)"

        elif (
            key == "Test ECE"
            or key == "Corruption Accuracy"
            or key == "Prediction Disagreement"
        ):
            if key == "Test ECE" or key == "Prediction Disagreement":
                result_p, result_strength = hypothesis_test(
                    baseline_dict[key], comapre_dict[key], alternative="less"
                )
                saftey_data[key] = result_p
                saftey_results[key] = [result_p, result_strength]
            elif key == "Corruption Accuracy":
                result_p, result_strength = hypothesis_test(
                    baseline_dict[key], comapre_dict[key], alternative="greater"
                )
                saftey_data[key] = result_p
                saftey_results[key] = [result_p, result_strength]
        else:
            result_p, result_strength = hypothesis_test(
                baseline_dict[key], comapre_dict[key], alternative="greater"
            )
            sharp_data[key] = result_p
            sharp_results[key] = [result_p, result_strength]
    new_pvalues_saftey = scipy.stats.false_discovery_control(
        list(saftey_data.values()), method="by"
    )
    new_pvalues_sharp = scipy.stats.false_discovery_control(
        list(sharp_data.values()), method="by"
    )
    print(new_pvalues_saftey)
    print(saftey_data)
    for p_idx, key in enumerate(saftey_data):
        print(p_idx)
        print("results")
        print(saftey_results[key])
        if new_pvalues_saftey[p_idx] < alpha:
            table_data[key] = f"\\cmark ({saftey_results[key][1]})"
        else:
            table_data[key] = "\\xmark (N/A)"
    for p_idx, key in enumerate(sharp_data):
        if new_pvalues_sharp[p_idx] < alpha:
            table_data[key] = f"\\cmark ({sharp_results[key][1]})"
        else:
            table_data[key] = "\\xmark (N/A)"
    return table_data


def hypothesis_tables_imagenet(baseline_dict, comapre_dict, alpha=0.05):
    table_data = {}
    saftey_data = {}
    sharp_data = {}
    saftey_results = {}
    sharp_results = {}
    for key in baseline_dict.keys():
        if key == "Generalisation Gap":
            test_results = hypothesis_test(
                baseline_dict[key], comapre_dict[key], alternative="less"
            )
            if test_results[0] < alpha:
                table_data[key] = f"\\cmark ({test_results[1]})"
            else:
                table_data[key] = f"\\xmark (N/A)"
        elif key == "Test Accuracy":
            test_results = hypothesis_test(
                baseline_dict[key], comapre_dict[key], alternative="greater"
            )
            if test_results[0] < alpha:
                table_data[key] = f"\\cmark ({test_results[1]})"
            else:
                table_data[key] = f"\\xmark (N/A)"

        elif (
            key == "Test ECE"
            or key == "Corruption Accuracy"
            or key == "Prediction Disagreement"
        ):
            if key == "Test ECE" or key == "Prediction Disagreement":
                result_p, result_strength = hypothesis_test(
                    baseline_dict[key], comapre_dict[key], alternative="less"
                )
                saftey_data[key] = result_p
                saftey_results[key] = [result_p, result_strength]
            elif key == "Corruption Accuracy":
                result_p, result_strength = hypothesis_test(
                    baseline_dict[key], comapre_dict[key], alternative="greater"
                )
                saftey_data[key] = result_p
                saftey_results[key] = [result_p, result_strength]
        else:
            result_p, result_strength = hypothesis_test(
                baseline_dict[key], comapre_dict[key], alternative="greater"
            )
            sharp_data[key] = result_p
            sharp_results[key] = [result_p, result_strength]
    new_pvalues_saftey = scipy.stats.false_discovery_control(
        list(saftey_data.values()), method="by"
    )
    new_pvalues_sharp = scipy.stats.false_discovery_control(
        list(sharp_data.values()), method="by"
    )
    print(new_pvalues_saftey)
    print(saftey_data)
    for p_idx, key in enumerate(saftey_data):
        print(p_idx)
        print("results")
        print(saftey_results[key])
        if new_pvalues_saftey[p_idx] < alpha:
            table_data[key] = f"\\cmark ({saftey_results[key][1]})"
        else:
            table_data[key] = "\\xmark (N/A)"
    for p_idx, key in enumerate(sharp_data):
        if new_pvalues_sharp[p_idx] < alpha:
            table_data[key] = f"\\cmark ({sharp_results[key][1]})"
        else:
            table_data[key] = "\\xmark (N/A)"
    return table_data


def significance_test_cifar(dataset, architecture):
    base_df, base_SAM_df, aug_df, aug_SAM_df, wd_df, wd_SAM_df = (
        create_processed_table_cifar(dataset, architecture)
    )
    base_SAM = hypothesis_tables_cifar(base_df, base_SAM_df, alpha=0.05)
    aug = hypothesis_tables_cifar(base_df, aug_df, alpha=0.05)
    aug_SAM = hypothesis_tables_cifar(base_df, aug_SAM_df, alpha=0.05)
    wd = hypothesis_tables_cifar(base_df, wd_df, alpha=0.05)
    wd_SAM = hypothesis_tables_cifar(base_df, wd_SAM_df, alpha=0.05)
    data = [base_SAM, aug, aug_SAM, wd, wd_SAM]
    df = pd.DataFrame(data)
    df.to_csv(f"./models/{dataset}/{architecture}_significance.csv")
    return df


def significance_test_cifar_SAM(architecture):
    (
        base_df,
        base_SAM_5,
        base_SAM_25,
        base_SAM_05,
        base_SAM_025,
        base_SAM_005,
        base_SAM_0025,
    ) = create_processed_table_SAM(architecture)
    base_SAM_5 = hypothesis_tables_cifar(base_df, base_SAM_5, alpha=0.05)
    base_SAM_25 = hypothesis_tables_cifar(base_df, base_SAM_25, alpha=0.05)
    base_SAM_05 = hypothesis_tables_cifar(base_df, base_SAM_05, alpha=0.05)
    base_SAM_025 = hypothesis_tables_cifar(base_df, base_SAM_025, alpha=0.05)
    base_SAM_005 = hypothesis_tables_cifar(base_df, base_SAM_005, alpha=0.05)
    base_SAM_0025 = hypothesis_tables_cifar(base_df, base_SAM_0025, alpha=0.05)
    data = [
        base_SAM_5,
        base_SAM_25,
        base_SAM_05,
        base_SAM_025,
        base_SAM_005,
        base_SAM_0025,
    ]
    df = pd.DataFrame(data)
    df.to_csv(f"./models/{architecture}_rho_SAM_significance.csv")
    return df


def significance_test_imagenet(dataset, architecture):
    base_df, base_SAM_df, aug_df, aug_SAM_df, wd_df, wd_SAM_df = (
        create_processed_table_imagenet(dataset, architecture)
    )
    base_SAM = hypothesis_tables_imagenet(base_df, base_SAM_df, alpha=0.05)
    aug = hypothesis_tables_imagenet(base_df, aug_df, alpha=0.05)
    aug_SAM = hypothesis_tables_imagenet(base_df, aug_SAM_df, alpha=0.05)
    wd = hypothesis_tables_imagenet(base_df, wd_df, alpha=0.05)
    wd_SAM = hypothesis_tables_imagenet(base_df, wd_SAM_df, alpha=0.05)
    data = [base_SAM, aug, aug_SAM, wd, wd_SAM]
    df = pd.DataFrame(data)
    df.to_csv(f"./models/{dataset}/{architecture}_significance.csv")
    return df


def main():
    os.chdir(
        "/cephfs/volumes/hpc_data_prj/inf_func_transition_nnt/30e6dc11-d145-4945-be2d-103e810d4f0a/sharpness_diversity/"
    )
    significance_test_cifar("CIFAR10", "ResNet18")
    significance_test_cifar("CIFAR10", "VGG19")
    significance_test_cifar("CIFAR10", "ViT")


if __name__ == "__main__":
    main()
