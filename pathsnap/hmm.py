
import numpy as np
import pandas as pd


def transition_score(
    previous_arc,
    current_arc,
    starts,
    ends,
    node_xy,
    transition_scale=30.0
):
    """Approximate road-transition plausibility."""

    previous_arc = int(previous_arc)
    current_arc = int(current_arc)

    if previous_arc == current_arc:
        return 0.0

    previous_end = int(ends[previous_arc])
    current_start = int(starts[current_arc])

    # Direct connection between directed road arcs
    if previous_end == current_start:
        return 0.0

    # Approximate gap between consecutive road arcs.
    # This is NOT a network shortest-path distance.
    gap = np.linalg.norm(
        node_xy[previous_end] - node_xy[current_start]
    )

    return -float(gap / transition_scale)


def viterbi_decode(
    candidates,
    arcs,
    node_xy,
    probability_col="candidate_probability",
    gap_reset_seconds=30.0,
    transition_weight=1.0,
    transition_mode="straight",
    network_cutoff_m=300.0,
):
    """
    Decode one trip's candidate road sequence.

    candidates must contain:
        observation_id, arc_id, timestamp,
        candidate_probability

    Returns one predicted arc per GPS observation.
    """

    required = {
        "observation_id",
        "arc_id",
        "timestamp",
        probability_col
    }

    if not required.issubset(candidates.columns):
        raise ValueError(
            f"Missing columns: {required - set(candidates.columns)}"
        )

    starts = arcs["from_node"].to_numpy(dtype=int)
    ends = arcs["to_node"].to_numpy(dtype=int)

    if transition_mode not in ("straight", "network"):
        raise ValueError("Invalid transition_mode")

    network_distance = None

    if transition_mode == "network":
        from pathsnap.road_graph import (
            build_road_graph,
            make_cached_distance,
        )
        graph = build_road_graph(arcs, node_xy)
        network_distance = make_cached_distance(
            graph, cutoff_m=network_cutoff_m
        )

    observations = [
        group.reset_index(drop=True)
        for _, group in candidates.groupby(
            "observation_id", sort=True
        )
    ]

    if not observations:
        return pd.DataFrame(
            columns=["observation_id", "predicted_arc_id"]
        )

    scores = []
    backpointers = []

    for t, group in enumerate(observations):
        probabilities = group[
            probability_col
        ].to_numpy(dtype=float)

        probabilities = np.clip(
            probabilities, 1e-12, 1.0
        )
        probabilities /= probabilities.sum()

        emissions = np.log(probabilities)

        if t == 0:
            scores.append(emissions)
            backpointers.append(
                np.full(len(group), -1, dtype=int)
            )
            continue

        previous = observations[t - 1]

        time_gap = float(
            group["timestamp"].iloc[0]
            - previous["timestamp"].iloc[0]
        )

        # Reset sequence after very large GPS gaps.
        if (
            time_gap <= 0
            or time_gap > gap_reset_seconds
        ):
            scores.append(emissions)
            backpointers.append(
                np.full(len(group), -1, dtype=int)
            )
            continue

        previous_scores = scores[-1]
        current_scores = np.full(
            len(group), -np.inf
        )
        current_back = np.full(
            len(group), -1, dtype=int
        )

        for j, current_arc in enumerate(
            group["arc_id"]
        ):
            best_score = -np.inf
            best_index = -1

            for k, previous_arc in enumerate(
                previous["arc_id"]
            ):
                if transition_mode == "network":
                    previous_arc = int(previous_arc)
                    current_arc = int(current_arc)

                    if previous_arc == current_arc:
                        transition = 0.0
                    else:
                        source = int(ends[previous_arc])
                        target = int(starts[current_arc])

                        distance = network_distance(source, target)

                        if not np.isfinite(distance):
                            transition = -20.0
                        else:
                            transition = -distance / 30.0
                else:
                    transition = transition_score(
                        previous_arc,
                        current_arc,
                        starts,
                        ends,
                        node_xy
                    )

                total = (
                    previous_scores[k]
                    + transition_weight * transition
                    + emissions[j]
                )

                if total > best_score:
                    best_score = total
                    best_index = k

            current_scores[j] = best_score
            current_back[j] = best_index

        scores.append(current_scores)
        backpointers.append(current_back)

    # Trace backward within each continuous sequence.
    selected = np.zeros(len(observations), dtype=int)

    segment_end = len(observations) - 1

    while segment_end >= 0:
        position = segment_end
        selected[position] = int(
            np.argmax(scores[position])
        )

        while (
            position > 0
            and backpointers[position][
                selected[position]
            ] != -1
        ):
            previous_choice = int(
                backpointers[position][
                    selected[position]
                ]
            )
            position -= 1
            selected[position] = previous_choice

        segment_end = position - 1

    result = []

    for group, index in zip(observations, selected):
        result.append({
            "observation_id": int(
                group["observation_id"].iloc[0]
            ),
            "predicted_arc_id": int(
                group["arc_id"].iloc[index]
            )
        })

    return pd.DataFrame(result)
