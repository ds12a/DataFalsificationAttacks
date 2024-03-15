import numpy as np


def additive(col, margin, pctVar=0.1):
    toAdd = np.random.uniform(margin * (1 - pctVar), margin * (1 + pctVar), col.shape)

    return col + toAdd


def deductive(col, margin, pctVar=0.1):
    modified = additive(col, -margin, pctVar)
    modified[modified < 0] = 0

    return modified


def movingWeightedAverageAdditive(col, margin, pctVar=0.1, window=10):
    modified = col.copy()

    weights = (np.arange(window)[::-1] + 1) ** 5
    movingAvgs = np.convolve(col, weights, "valid") / np.sum(weights)
    startup = np.arange(window - 1) * margin / window

    modified[: window - 1] += startup
    modified[window - 1 :] = additive(movingAvgs, margin, pctVar)

    return modified


def theoreticalPower (data):
    # Constants
    rho = 1.225  # 1.23 # Air density (kg/m^3)
    C_p = 0.59  # Turbine power coefficent
    A = 6362  # 20106.192983 # Rotor swept area (m^2)

    thd = 0.015

    max_power = 3618.73291015625
    turbine_direction = 205  # degs

    computed_theoretical_power = (
        data["Wind Speed (m/s)"] ** 3 * C_p * A * rho / 2 / 1000
    )  # Convert W to kW

    computed_theoretical_power_dir = computed_theoretical_power * np.abs(
        np.cos(np.deg2rad(data["Wind Direction (°)"] - turbine_direction))
    )  # Account for wind dir
    computed_theoretical_power = computed_theoretical_power.clip(upper=max_power)

    return computed_theoretical_power


def cyclicPhysics(data, m_p, f):
    computed_theoretical_power = theoreticalPower(data)

    efficency = np.average(
        (data["Theoretical_Power_Curve (KW)"] / computed_theoretical_power).dropna()
    )
    # computed_theoretical_power_with_efficency = computed_theoretical_power * efficency
    # print(f"efficency: {efficency}")

    computed_measured_power = computed_theoretical_power + np.random.normal(
        0, 5, len(computed_theoretical_power)
    )

    falsified_power = (
        computed_measured_power
        + (m_p * np.cos(np.arange(0, len(computed_measured_power) * 10, 10) * f))
        * np.random.random()
    ).clip(lower=0)

    return falsified_power


# def camouflageAdditive(col, margin, pctVar=0.1):
#     modified = col.copy()

#     modified[np.argsort(modified)] += np.sort(
#         np.random.uniform(margin * (1 - pctVar), margin * (1 + pctVar), modified.shape)
#     )[::-1]

#     return modified


# def camouflageDeductive(col, margin, pctVar=0.1):
#     modified = col.copy()

#     modified[np.argsort(modified)] -= np.sort(
#         np.random.uniform(margin * (1 - pctVar), margin * (1 + pctVar), modified.shape)
#     )
#     modified[modified < 0] = 0

#     return modified
