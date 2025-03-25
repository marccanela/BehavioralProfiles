import itertools
import pickle as pkl
import random
from collections import Counter
from math import pi

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import null_space
from scipy.special import erf
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests


def get_syllable_list(file, total_bins, bin_num):

    df = pd.read_csv(file)
    syllable_list = df["syllable"].tolist()
    bin_length = int(len(syllable_list) / total_bins)
    syllables = [
        syllable_list[i : i + bin_length]
        for i in range(0, len(syllable_list), bin_length)
    ]
    syllables = syllables[bin_num]

    return syllables


def filter_data(specifics, directory, total_bins, bin_num, target_values):

    # Filter based on specifics
    for column, specific in specifics.items():
        target_values = target_values[target_values[column] == specific]
    target_values = list(target_values.experiment_id)

    # Create a syllable dict
    syllables_dict = {}
    for file in directory.iterdir():
        if file.suffix == ".csv":

            # Filter based on conditions
            for value in target_values:
                if value in file.stem:

                    # Get syllables
                    syllables = get_syllable_list(file, total_bins, bin_num)
                    syllables_dict[value] = syllables

    return syllables_dict


def create_transition_dict(syllables_dict, silence_diagonal=False, normalize=False):

    transition_dict = {}

    # Find the maximum number of unique states across all the tags
    states = []
    for tag, syllables_list in syllables_dict.items():
        unique_states = list(set(syllables_list))
        states.extend(unique_states)
    states = list(set(states))

    for tag, syllables_list in syllables_dict.items():

        # Create an empty transition matrix filled with zeros
        transition_matrix = np.zeros((len(states), len(states)))

        # Iterate over each state's list of values and update the matrix
        for i in range(len(syllables_list) - 1):
            if syllables_list[i] in states and syllables_list[i + 1] in states:
                current_state_index = states.index(syllables_list[i])
                next_state_index = states.index(syllables_list[i + 1])
                transition_matrix[current_state_index][next_state_index] += 1

        # Silence diagonal
        if silence_diagonal:
            np.fill_diagonal(transition_matrix, 0)

        # Calculate row_sums and perform division
        if normalize:
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(
                transition_matrix, row_sums, where=row_sums != 0
            )

        # Add the transition matrix to the dictionary
        transition_dict[tag] = transition_matrix

    return transition_dict, states


def filter_syllables_dict(syllables_dict, states, target_values, column):

    ids = []
    for combination in states:
        booleans = target_values[column] == combination
        filtered_values = target_values.loc[booleans, "experiment_id"]
        ids.extend(filtered_values.tolist())

    filtered_dict = {
        id: syllables_dict[id] for id in ids if id in syllables_dict.keys()
    }

    return filtered_dict


def compute_stationary_distributions(transition_dict, states):

    transmats = list(transition_dict.values())
    stacked = np.dstack(transmats)
    combined = np.sum(stacked, axis=2)

    zero_rows = np.all(combined == 0, axis=1)
    new_states = [state for state, zero in zip(states, zero_rows) if not zero]
    new_matrix = combined[~zero_rows, :][:, ~zero_rows]

    row_sums = new_matrix.sum(axis=1, keepdims=True)
    new_matrix = np.divide(new_matrix, row_sums)

    stationary = np.linalg.matrix_power(new_matrix, 1000)
    stationary_distribution = stationary[0, :]

    return stationary_distribution, new_states


def plot_stationary_distribution(stationary_dict):

    all_states = []
    for state in stationary_dict.values():
        all_states.extend(state[1])
    states = list(set(all_states))

    data = pd.DataFrame(index=stationary_dict.keys(), columns=states)

    for letter, value in stationary_dict.items():
        stationary_distribution = value[0]
        specific_states = value[1]
        for state, prob in zip(specific_states, stationary_distribution):
            data.loc[letter, state] = prob
    data = data.fillna(0)

    # Reset the index to make row names a column
    df_reset = data.reset_index()
    melted_df = pd.melt(
        df_reset, id_vars=["index"], var_name="Syllable", value_name="Probability"
    )
    melted_df = melted_df.rename(columns={"index": "Profiles"})

    sns.barplot(data=melted_df, x="Syllable", y="Probability", hue="Profiles")

    return melted_df


def manhattan_distance(transmats_A, transmats_B):

    A = list(transmats_A.values())
    B = list(transmats_B.values())

    # Combine the matrices (axis=2 will stack them along the 3rd dimension)
    a = np.dstack(A)
    b = np.dstack(B)

    # Calculate the mean along the 3rd dimension, ignoring NaN values
    a_mean = np.nanmean(a, axis=2)
    b_mean = np.nanmean(b, axis=2)

    # Calculate the distance
    distance = np.nansum(np.abs(a_mean - b_mean))

    return distance


def global_stat(filtered_dict, combination, target_values, column):

    tags_A = target_values[target_values[column] == combination[0]].experiment_id
    transmats_A = {
        tag: filtered_dict[tag] for tag in tags_A if tag in filtered_dict.keys()
    }

    tags_B = target_values[target_values[column] == combination[1]].experiment_id
    transmats_B = {
        tag: filtered_dict[tag] for tag in tags_B if tag in filtered_dict.keys()
    }

    if len(transmats_A) == 0 or len(transmats_B) == 0:
        return None

    return manhattan_distance(transmats_A, transmats_B)


def plot_histogram(Tnull, Tobs, zscore, p_value, color="gray"):

    # Create the figure with a new size
    fig, ax = plt.subplots(figsize=(3, 2.5))  # Create both figure and axes

    # Plot histogram of Tnull
    sns.histplot(Tnull, bins=30, color=color, edgecolor="black", ax=ax)
    ax.axvline(Tobs, color="black", linestyle="--")  # Use ax for axvline
    ax.set_xlabel("Values", color="dimgray", fontsize=10, fontweight="bold")
    ax.set_ylabel("Frequency", color="dimgray", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", colors="dimgray", labelsize=8)
    ax.tick_params(axis="y", colors="dimgray", labelsize=8)

    # Remove the edge around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor("dimgray")

    # Print text information into the console
    print(f"z-score: {zscore:.2f}")
    print(f"p-value: {p_value:.2e}")

    # Adjust the size and layout of the figure
    plt.tight_layout()

    # Display the plot
    plt.show()


def shuffle_combined(target_values, column):

    # Deep copy the target_values
    new_target_values = target_values.copy()

    new_target_values[column] = (
        new_target_values[column].sample(frac=1).reset_index(drop=True)
    )

    return new_target_values


def filter_empty_transitions(transition_dict):
    return {
        tag: transition
        for tag, transition in transition_dict.items()
        if np.sum(transition) != 0
    }


def behavior_bootstrap(transition_dict, target_values, column, pvalue="gaussian"):

    # Filter the transition_dict to remove any empty transitions
    filtered_dict = filter_empty_transitions(transition_dict)

    # Get the unique combinations of the target_values
    uniques = list(set(target_values[column]))
    combinations = list(itertools.combinations(uniques, 2))

    # Compute the global test statistic
    Tobs = sum(
        global_stat(filtered_dict, combination, target_values, column)
        for combination in combinations
    )

    # Apply the bootstrap
    Tnull = []
    for i in range(10**3):
        shuffled_target = shuffle_combined(target_values, column)
        Tnull.append(
            sum(
                global_stat(filtered_dict, combination, shuffled_target, column)
                for combination in combinations
            )
        )

    # Gaussian approximation of the p-value
    if pvalue == "gaussian":
        zscore = (Tobs - np.mean(Tnull)) / np.std(Tnull)
        p_value = (1 - erf(zscore / np.sqrt(2))) / 2

    else:
        p_value = np.sum(Tnull > Tobs) / len(Tnull)
        zscore = None

    # Plot the histogram
    plot_histogram(Tnull, Tobs, zscore, p_value)


def pairwise_behavior_bootstrap(
    transition_dict, target_values, column, pvalue="gaussian"
):

    # Filter the transition_dict to remove any empty transitions
    filtered_dict = filter_empty_transitions(transition_dict)

    # Get the unique combinations of the target_values
    uniques = list(set(target_values[column]))
    combinations = list(itertools.combinations(uniques, 2))

    # Compute the individual global test statistics
    T_individual = {
        combination: global_stat(filtered_dict, combination, target_values, column)
        for combination in combinations
    }

    # Apply the bootstrap
    T_null_indiv = {}
    for combination in combinations:
        Tnull = []
        for i in range(10**3):
            shuffled_target = shuffle_combined(target_values, column)
            Tnull.append(
                global_stat(filtered_dict, combination, shuffled_target, column)
            )
        T_null_indiv[combination] = Tnull

    # Compute the p-values
    stats = []

    for combination in combinations:
        Tobs = T_individual[combination]
        Tnull = T_null_indiv[combination]

        if pvalue == "gaussian":
            zscore = (Tobs - np.mean(Tnull)) / np.std(Tnull)
            p_value = (1 - erf(zscore / np.sqrt(2))) / 2
        else:
            p_value = np.sum(Tnull > Tobs) / len(Tnull)
            zscore = None

        stats.append(
            {
                "Combination": combination,
                "Tobs": Tobs,
                "zscore": zscore,
                "p_value": p_value,
            }
        )

    # Create a DataFrame from the stats list
    stats_df = pd.DataFrame(stats)

    # Split the 'Combination' column into two separate columns
    stats_df["Combination"] = stats_df["Combination"].astype(str)
    stats_df[["Group1", "Group2"]] = (
        stats_df["Combination"].str.strip("()").str.split(", ", expand=True)
    )
    stats_df = stats_df.drop(columns=["Combination"])
    stats_df = stats_df[["Group1", "Group2", "Tobs", "zscore", "p_value"]]
    stats_df["Group1"] = stats_df["Group1"].str.strip("'")
    stats_df["Group2"] = stats_df["Group2"].str.strip("'")

    # Perform BH correction
    p_values = stats_df["p_value"].values
    _, padj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
    stats_df["padj"] = padj

    # Filter for significant combinations with padj <= 0.05
    stats_df["sign"] = stats_df["padj"] <= 0.05

    return stats_df


def explain_with_deepof(
    stationary_dict,
    syllables_dict,
    deepof,
    target_values,
    specific_combinations,
    specifics,
    total_bins,
    bin_num,
    column,
):

    # Filter specifics (direct/mediated)
    for mycolumn, specific in specifics.items():
        target_values = target_values[target_values[mycolumn] == specific]

    letter_features_dict = {}
    for letter, list_combinations in specific_combinations.items():

        # Identify the IDs of the corresponding letters
        ids = []
        for combination in list_combinations:
            booleans = target_values[column] == combination
            filtered_values = target_values.loc[booleans, "experiment_id"]
            ids.extend(filtered_values.tolist())

        # extract deepof data
        letter_deepof = {id: deepof[id] for id in ids if id in deepof.keys()}

        # Filter by time bin
        cut_deepof = {}
        for id, table in letter_deepof.items():
            if not table.empty:
                bin_size = len(table) // total_bins
                start_index = bin_num * bin_size + bin_num  # Start from 0
                end_index = start_index + bin_size
                cut_table = table.iloc[start_index:end_index]
                cut_deepof[id] = cut_table

        # Add the syllables
        filtered_syllables = {
            id: syllables_dict[id] for id in ids if id in syllables_dict.keys()
        }
        if len(filtered_syllables) != len(letter_deepof):
            print("Error with lengths for letter {letter}. Skipping problematic IDs.")
            continue

        deepof_syllables = {}
        for id, table in cut_deepof.items():
            if id in filtered_syllables.keys():
                newtable = table.copy()
                syllables_for_table = syllables_dict[id]
                if len(syllables_for_table) == len(newtable):
                    newtable["syllable"] = syllables_for_table
                    deepof_syllables[id] = newtable
                else:
                    print("Mismatch in syllable length for ID {id}. Skipping.")

        # Concatenate all animals in one table
        supertable = pd.concat(deepof_syllables.values(), ignore_index=True)
        supertable = supertable.dropna()

        # Compute mean features by syllable
        features_list = ["climbing", "sniffing", "huddle", "lookaround", "speed"]
        features = {
            feature: supertable.groupby("syllable")[feature].mean().to_dict()
            for feature in features_list
            if feature in supertable.columns
        }

        # Multiply the mean by the cluster's weight and add up
        ponderations = stationary_dict[letter]
        pond_dict = dict(zip(ponderations[1], ponderations[0]))

        summary_features = {}
        for feature, values in features.items():
            common_keys = set(values.keys()).intersection(pond_dict.keys())
            numerator = sum(values[key] * pond_dict[key] for key in common_keys)
            denominator = sum(pond_dict[key] for key in common_keys)
            ponderated_mean = numerator / denominator if denominator != 0 else 0
            summary_features[feature] = ponderated_mean

        letter_features_dict[letter] = summary_features

    return letter_features_dict


def plot_deepof_explanation(letter_features_dict):

    # Prepare data for plotting
    df = pd.DataFrame.from_dict(letter_features_dict, orient="index").reset_index()
    df = df.melt(id_vars="index", var_name="behavior", value_name="Values")
    df.rename(columns={"index": "Profiles"}, inplace=True)

    # Plotting
    g = sns.FacetGrid(
        df, col="behavior", sharey=False
    )  # Different y-axis for each behavior
    g.map(sns.barplot, "Profiles", "Values")
    g.set_titles("{col_name}")
    plt.show()


def radar_plot(letter_features_dict, profile, color):

    # Set data
    df = pd.DataFrame(letter_features_dict).T.reset_index()
    df.rename(columns={"index": "profiles"}, inplace=True)
    df["profiles"] = df["profiles"].str.upper()

    # number of variables
    categories = ["climbing", "huddle", "lookaround", "sniffing", "speed"]
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    row = df[df["profiles"] == profile]
    values = row.drop(columns="profiles").values.flatten().tolist()

    # Define scales of each variable
    scales = [(0, 0.4), (0, 0.4), (0.05, 0.55), (0.04, 0.12), (1, 8)]

    normalized_values = [
        (val - scale[0]) / (scale[1] - scale[0]) for val, scale in zip(values, scales)
    ]
    normalized_values += normalized_values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw={"polar": True})

    # Draw the axes and category labels
    plt.xticks(angles[:-1], categories, color="dimgray", size=8)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Adjust radial ticks to show normalized values
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", ""], color="dimgray", size=8
    )
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, normalized_values, linewidth=2, linestyle="solid", color="dimgray")
    ax.fill(angles, normalized_values, color=color)

    # Tidy up the layout
    plt.tight_layout()
    plt.show()


def dual_radar_plot(letter_features_dicts, profile, colors):

    # number of variables
    categories = ["climbing", "huddle", "lookaround", "sniffing", "speed"]
    scales = [(0, 0.4), (0, 0.4), (0.05, 0.55), (0.04, 0.12), (1, 8)]

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # ------- PART 1: Create background
    fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw={"polar": True})
    plt.xticks(angles[:-1], categories, color="dimgray", fontsize=10, fontweight="bold")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", ""], color="dimgray", size=8
    )
    plt.ylim(0, 1)

    # ------- PART 2: Add plots
    polygons = {}
    for time, letter_features_dict in letter_features_dicts.items():
        df = pd.DataFrame(letter_features_dict).T.reset_index()
        df.rename(columns={"index": "profiles"}, inplace=True)
        df["profiles"] = df["profiles"].str.upper()
        row = df[df["profiles"] == profile]
        values = row.drop(columns="profiles").values.flatten().tolist()

        normalized_values = [
            (val - scale[0]) / (scale[1] - scale[0])
            for val, scale in zip(values, scales)
        ]
        polygons[time] = normalized_values
        normalized_values += normalized_values[:1]
        ax.plot(
            angles,
            normalized_values,
            linewidth=2,
            linestyle="solid",
            alpha=1,
            color="dimgray",
        )
        ax.fill(angles, normalized_values, color=colors[time], alpha=0.5)

    # Tidy up the layout
    plt.tight_layout()
    plt.show()

    return polygons


def dual_syllable_radar_plot(dict_time_syllables, times, colors, max_lim=1):

    dict1 = dict(
        zip(dict_time_syllables[times[0]][1], dict_time_syllables[times[0]][0])
    )
    dict2 = dict(
        zip(dict_time_syllables[times[1]][1], dict_time_syllables[times[1]][0])
    )

    # Delete low abundance syllables
    dict1 = {key: value for key, value in dict1.items() if value >= 0.05}
    dict2 = {key: value for key, value in dict2.items() if value >= 0.05}

    # Harmonize the dictionaries
    all_keys = set(dict1.keys()).union(dict2.keys())
    for key in all_keys:
        if key not in dict1:
            dict1[key] = 0
        if key not in dict2:
            dict2[key] = 0

    dict1 = dict(sorted(dict1.items()))
    dict2 = dict(sorted(dict2.items()))

    # Create angles
    categories = list(dict1.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # ------- PART 1: Create background
    fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw={"polar": True})
    plt.xticks(angles[:-1], categories, color="dimgray", fontsize=10, fontweight="bold")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks(color="dimgray", size=8)
    plt.ylim(0, max_lim)

    # ------- PART 2: Add plots
    polygons = {}
    for time, letter_features_dict in zip(times, [dict1, dict2]):

        normalized_values = list(letter_features_dict.values())

        polygons[time] = normalized_values
        normalized_values += normalized_values[:1]
        ax.plot(
            angles,
            normalized_values,
            linewidth=2,
            linestyle="solid",
            alpha=1,
            color="dimgray",
        )
        ax.fill(angles, normalized_values, color=colors[time], alpha=0.5)

    # Tidy up the layout
    plt.tight_layout()
    plt.show()

    return polygons


def compute_iou(polygons):

    angles = [n / float(5) * 2 * pi for n in range(5)]

    polys = []
    for values in polygons.values():
        x_coords = [r * np.cos(theta) for r, theta in zip(values, angles)]
        y_coords = [r * np.sin(theta) for r, theta in zip(values, angles)]
        coords = list(zip(x_coords, y_coords))
        poly = Polygon(coords)
        polys.append(poly)

    poly1 = Polygon(polys[0])
    poly2 = Polygon(polys[1])

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0  # Invalid polygons result in IoU of 0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    return intersection / union if union > 0 else 0.0


def process_deepof(
    deepof, times, target_values, specifics_transmatrix, myspecific, categories, bins
):

    # Filter specifics (direct/mediated)
    for mycolumn, specific in specifics_transmatrix.items():
        target_values = target_values[target_values[mycolumn] == specific]

    # Identify the IDs of the corresponding letters
    booleans = target_values.group == myspecific
    filtered_values = target_values.loc[booleans, "experiment_id"]
    ids = filtered_values.tolist()

    # extract deepof data
    letter_deepof = {id: deepof[id] for id in ids if id in deepof.keys()}

    # Filter by time bin
    time_features_dicts = {}
    for bin_num in times:
        cut_deepof = {}
        for id, table in letter_deepof.items():
            if not table.empty:
                bin_size = len(table) // bins
                start_index = bin_num * bin_size + bin_num  # Start from 0
                end_index = start_index + bin_size
                cut_table = table.iloc[start_index:end_index]
                cut_deepof[id] = cut_table

        dict_features = {}
        for category in categories:
            averages = [table[category].mean() for table in cut_deepof.values()]
            dict_features[category] = averages

        time_features_dicts[bin_num] = dict_features

    return time_features_dicts


def dual_radar_plot_deepof(
    deepof,
    times,
    target_values,
    specifics_transmatrix,
    myspecific,
    colors,
    categories,
    scales,
    bins,
):

    all_data_dict = process_deepof(
        deepof,
        times,
        target_values,
        specifics_transmatrix,
        myspecific,
        categories,
        bins,
    )
    time_features_dicts = {}
    for bin_num, dict_features in all_data_dict.items():
        time_features_dicts[bin_num] = {
            category: np.mean(dict_features[category]) for category in categories
        }

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # ------- PART 1: Create background
    fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw={"polar": True})
    plt.xticks(angles[:-1], categories, color="dimgray", fontsize=10, fontweight="bold")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", ""], color="dimgray", size=8
    )
    plt.ylim(0, 1)

    # ------- PART 2: Add plots
    polygons = {}
    for time, letter_features_dict in time_features_dicts.items():
        values = list(letter_features_dict.values())

        normalized_values = [
            (val - scale[0]) / (scale[1] - scale[0])
            for val, scale in zip(values, scales)
        ]
        polygons[time] = normalized_values
        normalized_values += normalized_values[:1]
        ax.plot(
            angles,
            normalized_values,
            linewidth=2,
            linestyle="solid",
            alpha=1,
            color="dimgray",
        )
        ax.fill(angles, normalized_values, color=colors[time], alpha=0.5)

    # Tidy up the layout
    plt.tight_layout()
    plt.show()

    return polygons


def timelapse_deepof_radar_permutation(
    deepof,
    times,
    target_values,
    specifics_transmatrix,
    myspecific,
    categories,
    scales,
    color,
    bins,
    pvalue="gaussian",
):

    # Compute the global test statistic
    all_data_dict = process_deepof(
        deepof,
        times,
        target_values,
        specifics_transmatrix,
        myspecific,
        categories,
        bins,
    )
    time_features_dicts = {}
    for bin_num, dict_features in all_data_dict.items():
        time_features_dicts[bin_num] = {
            category: np.mean(dict_features[category]) for category in categories
        }

    polygons = {}
    for time, letter_features_dict in time_features_dicts.items():
        values = list(letter_features_dict.values())

        normalized_values = [
            (val - scale[0]) / (scale[1] - scale[0])
            for val, scale in zip(values, scales)
        ]
        polygons[time] = normalized_values

    Tobs = compute_iou(polygons)

    # Apply the bootstrap
    Tnull = []
    for i in range(10**4):

        time_features_dicts_bootstrap = {times[0]: {}, times[1]: {}}
        for category in categories:
            category_list = []
            for time in times:
                category_list.extend(all_data_dict[time][category])
            random.shuffle(category_list)
            mid = len(category_list) // 2
            time_features_dicts_bootstrap[times[0]][category] = np.mean(
                category_list[:mid]
            )
            time_features_dicts_bootstrap[times[1]][category] = np.mean(
                category_list[mid:]
            )

        polygons = {}
        for time, letter_features_dict in time_features_dicts_bootstrap.items():
            values = list(letter_features_dict.values())

            normalized_values = [
                (val - scale[0]) / (scale[1] - scale[0])
                for val, scale in zip(values, scales)
            ]
            polygons[time] = normalized_values

        Tnull.append(compute_iou(polygons))

    # Gaussian approximation of the p-value
    if pvalue == "gaussian":
        zscore = (Tobs - np.mean(Tnull)) / np.std(Tnull)
        p_value = (1 - erf(zscore / np.sqrt(2))) / 2

    else:
        Tnull = np.array(Tnull)
        p_value = 2 * min(np.mean(Tnull <= Tobs), np.mean(Tnull >= Tobs))
        zscore = (Tobs - np.mean(Tnull)) / np.std(Tnull)

    # Plot the histogram
    plot_histogram(Tnull, Tobs, zscore, p_value, color)


def harmonize_matrices(transition_dicts):

    # Delete tags
    reduced_transition_dicts = {}
    for time, mylist in transition_dicts.items():
        mydict = mylist[0]  # len = 18 (one for each animal)
        states = mylist[1]  # len = 22 (one list of syllables)
        reduced_transition_dicts[time] = [list(mydict.values()), states]

    # Harmonize matrices
    # Step 1: Find the common states across all matrices
    common_states = set(
        reduced_transition_dicts[next(iter(reduced_transition_dicts))][1]
    )

    # Iterate over all the times and update the common states
    for time in reduced_transition_dicts:
        states = reduced_transition_dicts[time][1]
        common_states &= set(states)

    # Convert the set to a list
    common_states = list(common_states)

    # Step 2: Filter matrices to keep only common states
    new_dict = {}
    n = 1
    for time, mylist in reduced_transition_dicts.items():
        matrices = mylist[0]  # len = 18 (one for each animal)
        states = mylist[1]  # len = 22 (one list of syllables)

        indices = [states.index(state) for state in common_states]
        dummy_dict = {}
        for matrix in matrices:
            filtered_matrix = matrix[np.ix_(indices, indices)]
            dummy_dict[f"time_{n}"] = filtered_matrix
            n += 1
        new_dict[time] = [dummy_dict, common_states]

    combination = list(new_dict.keys())

    all_transmats = {}
    for time, transition_dict in new_dict.items():
        transmats = filter_empty_transitions(transition_dict[0])
        all_transmats[time] = transmats

    if (
        len(all_transmats[combination[0]]) == 0
        or len(all_transmats[combination[1]]) == 0
    ):
        return None

    return all_transmats, combination, common_states


def timelapse_behavior_bootstrap(transition_dicts, color, pvalue="gaussian"):

    # Harmonize matrices
    all_transmats, combination, _ = harmonize_matrices(transition_dicts)

    # Compute the global test statistic
    Tobs = manhattan_distance(
        all_transmats[combination[0]], all_transmats[combination[1]]
    )

    # Apply the bootstrap
    all_dummies = []
    transmats_for_random = {}
    for transmats in all_transmats.values():
        all_dummies.extend(list(transmats.keys()))
        transmats_for_random.update(transmats)

    Tnull = []
    for i in range(10**3):
        random.shuffle(all_dummies)

        tags_A = all_dummies[: len(all_dummies) // 2]
        transmats_A = {dummy: transmats_for_random[dummy] for dummy in tags_A}

        tags_B = all_dummies[len(all_dummies) // 2 :]
        transmats_B = {dummy: transmats_for_random[dummy] for dummy in tags_B}

        Tnull.append(manhattan_distance(transmats_A, transmats_B))

    # Gaussian approximation of the p-value
    if pvalue == "gaussian":
        zscore = (Tobs - np.mean(Tnull)) / np.std(Tnull)
        p_value = (1 - erf(zscore / np.sqrt(2))) / 2

    else:
        p_value = np.sum(Tnull > Tobs) / len(Tnull)
        zscore = None

    # Plot the histogram
    plot_histogram(Tnull, Tobs, zscore, p_value, color)

    return zscore, p_value


def compute_stationary_dict(specific_combinations, syllables_dict, target_values):

    stationary_dict = {}
    for letter, value in specific_combinations.items():
        filtered_dict = filter_syllables_dict(
            syllables_dict, value, target_values, "group"
        )
        transition_dict, states = create_transition_dict(
            filtered_dict, silence_diagonal=True, normalize=False
        )
        stationary, new_states = compute_stationary_distributions(
            transition_dict, states
        )
        stationary_dict[letter] = [stationary, new_states]

    return stationary_dict


def compute_stationary_distributions_animal(transition_dict, states):

    transmats = list(transition_dict.values())

    stationary_distributions = []
    new_states_list = []
    for transmat in transmats:
        zero_rows = np.all(transmat == 0, axis=1)
        new_states = [state for state, zero in zip(states, zero_rows) if not zero]
        new_matrix = transmat[~zero_rows, :][:, ~zero_rows]

        row_sums = new_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        new_matrix = np.divide(new_matrix, row_sums)

        stationary = np.linalg.matrix_power(new_matrix, 1000)
        stationary_distribution = stationary[0, :]

        stationary_distributions.append(stationary_distribution)
        new_states_list.append(new_states)

    return stationary_distributions, new_states_list


def timelapse_circular_transition_graph(transition_dicts, colors, ax=None):

    # Harmonize matrices
    all_transmats, combination, unique_states = harmonize_matrices(transition_dicts)

    # Compute mean transition matrices
    A = list(all_transmats[combination[0]].values())
    B = list(all_transmats[combination[1]].values())
    A_mean = np.nanmean(np.dstack(A), axis=2)
    B_mean = np.nanmean(np.dstack(B), axis=2)

    # Compute difference matrix
    delta = B_mean - A_mean

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(unique_states)))

    # Add weighted edges based on transition probabilities
    for i in range(len(unique_states)):
        for j in range(len(unique_states)):
            if i != j:  # Avoid self-transitions
                change = delta[i, j]
                if abs(change) >= 0.01:
                    color = colors[3] if change > 0 else colors[2]
                    width = abs(change) * 2
                    G.add_edge(i, j, weight=width, color=color)

    # Get edge attributes
    edges = G.edges()
    colors = [G[u][v]["color"] for u, v in edges]
    weights = [G[u][v]["weight"] for u, v in edges]

    # Draw the graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    pos = nx.circular_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        # node_size=node_size,
        node_color="snow",
        edgecolors="dimgray",
        font_weight="bold",
        arrows=False,
        edge_color=colors,
        width=weights,
        # width=1.5,
        alpha=0.8,
        node_size=250,
        font_size=10,
        font_color="dimgray",
        labels={i: state for i, state in enumerate(unique_states)},
    )

    return ax


def timelapse_circular_ind_transition_graph(transition_dicts, colors, time, ax=None):

    # Harmonize matrices
    all_transmats, _, unique_states = harmonize_matrices(transition_dicts)

    # Compute mean transition matrices
    A = list(all_transmats[time].values())
    A_mean = np.nanmean(np.dstack(A), axis=2)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(unique_states)))

    # Add weighted edges based on transition probabilities
    for i in range(len(unique_states)):
        for j in range(len(unique_states)):
            if i != j:  # Avoid self-transitions
                change = A_mean[i, j]
                if abs(change) >= 0.01:
                    color = colors[time]
                    width = abs(change) * 2
                    G.add_edge(i, j, weight=width, color=color)

    # Get edge attributes
    edges = G.edges()
    colors = [G[u][v]["color"] for u, v in edges]
    weights = [G[u][v]["weight"] for u, v in edges]

    # Draw the graph
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))

    pos = nx.circular_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        # node_size=node_size,
        node_color="snow",
        edgecolors="dimgray",
        font_weight="bold",
        arrows=False,
        edge_color=colors,
        width=weights,
        # width=1.5,
        # alpha=0.8,
        node_size=200,
        font_size=8,
        font_color="dimgray",
        labels={i: state for i, state in enumerate(unique_states)},
    )

    return ax


def timelapse_heatmap_transition_graph(transition_dicts, colors, period, vmax, ax=None):

    # Harmonize matrices
    all_transmats, combination, unique_states = harmonize_matrices(transition_dicts)

    # Compute mean transition matrices
    A = list(all_transmats[combination[0]].values())
    B = list(all_transmats[combination[1]].values())
    A_mean = np.nanmean(np.dstack(A), axis=2)
    B_mean = np.nanmean(np.dstack(B), axis=2)

    # Compute difference matrix
    if period == "OFF":
        delta = A_mean
    elif period == "ON":
        delta = B_mean

    # Colors
    if colors[2] == "lightblue":
        cmap = "Blues"
    elif colors[2] == "pink":
        cmap = "Reds"

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5))

    # Create a DataFrame for better labeling of the heatmap
    sns.heatmap(
        delta,
        cmap=cmap,
        # center=0,
        # vmin=-3,
        vmax=vmax,
        xticklabels=unique_states,
        yticklabels=unique_states,
    )

    # Plot the heatmap
    sns.set_theme(style="whitegrid")

    # Set labels and title
    plt.xlabel("Next State", loc="left")
    plt.ylabel("Current State", loc="top")

    # Grey color
    ax.xaxis.label.set_color("dimgray")
    ax.yaxis.label.set_color("dimgray")
    ax.tick_params(axis="x", colors="dimgray", labelsize=8)
    ax.tick_params(axis="y", colors="dimgray", labelsize=8)
    plt.xticks(rotation=0)

    return ax


def shannon_entropy_of_transitions(P):

    n = P.shape[0]
    A = P.T - np.eye(n)  # (P^T - I)π = 0
    A[-1, :] = 1  # Replace last row to enforce sum(π) = 1
    b = np.zeros(n)
    b[-1] = 1  # Constraint to normalize the probabilities
    pi = np.linalg.lstsq(A, b, rcond=None)[0]  # Solve for π
    entropy = -np.sum(pi[:, None] * P * np.log2(P, where=P > 0))

    return entropy


def analyze_transition_changes(transition_dicts):

    # Harmonize matrices
    all_transmats, combination, _ = harmonize_matrices(transition_dicts)

    # Compute mean transition matrices
    A = list(all_transmats[combination[0]].values())
    B = list(all_transmats[combination[1]].values())

    measures = {}

    # Count upregulated & downregulated transitions
    upregulated = []
    downregulated = []
    for matrix_A, matrix_B in zip(A, B):
        delta = matrix_B - matrix_A
        upregulated.append(np.sum(delta > 0.01))
        downregulated.append(np.sum(delta < -0.01))
    measures["upregulated"] = upregulated
    measures["downregulated"] = downregulated

    # Compute degree centrality
    measures["degree_A"] = [nx.degree_centrality(nx.DiGraph(matrix)) for matrix in A]
    measures["degree_B"] = [nx.degree_centrality(nx.DiGraph(matrix)) for matrix in B]

    # Compute entropy
    measures["entropy_A"] = [shannon_entropy_of_transitions(matrix) for matrix in A]
    measures["entropy_B"] = [shannon_entropy_of_transitions(matrix) for matrix in B]

    return measures
