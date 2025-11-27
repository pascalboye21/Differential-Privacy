import numpy as np
import pandas as pd
import logging
import os
import warnings
import matplotlib.patches as mpatches
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt
from exponential_mechanism import Exponential_Mechanism

min_0: float
min_2: float
max_0: float
max_2: float

cnames = {
    "aliceblue": "#F0F8FF",
    "antiquewhite": "#FAEBD7",
    "aqua": "#00FFFF",
    "aquamarine": "#7FFFD4",
    "azure": "#F0FFFF",
    "beige": "#F5F5DC",
    "bisque": "#FFE4C4",
    "black": "#000000",
    "blanchedalmond": "#FFEBCD",
    "blue": "#0000FF",
    "blueviolet": "#8A2BE2",
    "brown": "#A52A2A",
    "burlywood": "#DEB887",
    "cadetblue": "#5F9EA0",
    "chartreuse": "#7FFF00",
    "chocolate": "#D2691E",
    "coral": "#FF7F50",
    "cornflowerblue": "#6495ED",
    "cornsilk": "#FFF8DC",
    "crimson": "#DC143C",
    "cyan": "#00FFFF",
    "darkblue": "#00008B",
    "darkcyan": "#008B8B",
    "darkgoldenrod": "#B8860B",
    "darkgray": "#A9A9A9",
    "darkgreen": "#006400",
    "darkkhaki": "#BDB76B",
    "darkmagenta": "#8B008B",
    "darkolivegreen": "#556B2F",
    "darkorange": "#FF8C00",
    "darkorchid": "#9932CC",
    "darkred": "#8B0000",
    "darksalmon": "#E9967A",
    "darkseagreen": "#8FBC8F",
    "darkslateblue": "#483D8B",
    "darkslategray": "#2F4F4F",
    "darkturquoise": "#00CED1",
    "darkviolet": "#9400D3",
    "deeppink": "#FF1493",
    "deepskyblue": "#00BFFF",
    "dimgray": "#696969",
    "dodgerblue": "#1E90FF",
    "firebrick": "#B22222",
    "floralwhite": "#FFFAF0",
    "forestgreen": "#228B22",
    "fuchsia": "#FF00FF",
    "gainsboro": "#DCDCDC",
    "ghostwhite": "#F8F8FF",
    "gold": "#FFD700",
    "goldenrod": "#DAA520",
    "gray": "#808080",
    "green": "#008000",
    "greenyellow": "#ADFF2F",
    "honeydew": "#F0FFF0",
    "hotpink": "#FF69B4",
    "indianred": "#CD5C5C",
    "indigo": "#4B0082",
    "ivory": "#FFFFF0",
    "khaki": "#F0E68C",
    "lavender": "#E6E6FA",
    "lavenderblush": "#FFF0F5",
    "lawngreen": "#7CFC00",
    "lemonchiffon": "#FFFACD",
    "lightblue": "#ADD8E6",
    "lightcoral": "#F08080",
    "lightcyan": "#E0FFFF",
    "lightgoldenrodyellow": "#FAFAD2",
    "lightgreen": "#90EE90",
    "lightgray": "#D3D3D3",
    "lightpink": "#FFB6C1",
    "lightsalmon": "#FFA07A",
    "lightseagreen": "#20B2AA",
    "lightskyblue": "#87CEFA",
    "lightslategray": "#778899",
    "lightsteelblue": "#B0C4DE",
    "lightyellow": "#FFFFE0",
    "lime": "#00FF00",
    "limegreen": "#32CD32",
    "linen": "#FAF0E6",
    "magenta": "#FF00FF",
    "maroon": "#800000",
    "mediumaquamarine": "#66CDAA",
    "mediumblue": "#0000CD",
    "mediumorchid": "#BA55D3",
    "mediumpurple": "#9370DB",
    "mediumseagreen": "#3CB371",
    "mediumslateblue": "#7B68EE",
    "mediumspringgreen": "#00FA9A",
    "mediumturquoise": "#48D1CC",
    "mediumvioletred": "#C71585",
    "midnightblue": "#191970",
    "mintcream": "#F5FFFA",
    "mistyrose": "#FFE4E1",
    "moccasin": "#FFE4B5",
    "navajowhite": "#FFDEAD",
    "navy": "#000080",
    "oldlace": "#FDF5E6",
    "olive": "#808000",
    "olivedrab": "#6B8E23",
    "orange": "#FFA500",
    "orangered": "#FF4500",
    "orchid": "#DA70D6",
    "palegoldenrod": "#EEE8AA",
    "palegreen": "#98FB98",
    "paleturquoise": "#AFEEEE",
    "palevioletred": "#DB7093",
    "papayawhip": "#FFEFD5",
    "peachpuff": "#FFDAB9",
    "peru": "#CD853F",
    "pink": "#FFC0CB",
    "plum": "#DDA0DD",
    "powderblue": "#B0E0E6",
    "purple": "#800080",
    "red": "#FF0000",
    "rosybrown": "#BC8F8F",
    "royalblue": "#4169E1",
    "saddlebrown": "#8B4513",
    "salmon": "#FA8072",
    "sandybrown": "#FAA460",
    "seagreen": "#2E8B57",
    "seashell": "#FFF5EE",
    "sienna": "#A0522D",
    "silver": "#C0C0C0",
    "skyblue": "#87CEEB",
    "slateblue": "#6A5ACD",
    "slategray": "#708090",
    "snow": "#FFFAFA",
    "springgreen": "#00FF7F",
    "steelblue": "#4682B4",
    "tan": "#D2B48C",
    "teal": "#008080",
    "thistle": "#D8BFD8",
    "tomato": "#FF6347",
    "turquoise": "#40E0D0",
    "violet": "#EE82EE",
    "wheat": "#F5DEB3",
    "white": "#FFFFFF",
    "whitesmoke": "#F5F5F5",
    "yellow": "#FFFF00",
    "yellowgreen": "#9ACD32",
}


def weights_and_statistics_and_sensitivity_and_stats(df: pd.DataFrame):
    statistics = []
    weights = []
    sen = []
    stats = []

    stats.append(lambda x: np.mean(x[:, 0]))
    stats.append(lambda x: skew(list(x[:, 0])))
    stats.append(lambda x: kurtosis(list(x[:, 0])))
    stats.append(lambda x: np.mean(x[:, 2]))
    stats.append(lambda x: skew(list(x[:, 2])))
    stats.append(lambda x: kurtosis(list(x[:, 2])))
    stats.append(lambda x: np.mean(x[:, 2][x[:, 1] == True]))
    stats.append(lambda x: np.mean(x[:, 2][x[:, 1] == False]))
    stats.append(lambda x: np.cov(np.transpose(x.astype("float"))))

    global min_0
    global min_2
    global max_0
    global max_2
    min_0 = np.min(df.values[:, 0])
    min_2 = np.min(df.values[:, 2])
    max_0 = np.max(df.values[:, 0])
    max_2 = np.max(df.values[:, 2])

    l_1 = 6
    etas = []
    for i in range(l_1):
        etas.append(np.pi * i / l_1)

    l_2 = 50
    for eta in etas:
        for i in range(1, l_2):
            q = i / l_2
            quantile = np.quantile(
                np.cos(eta) * df.values[:, 0] + np.sin(eta) * df.values[:, 2], q
            )
            statistics.append(
                lambda x, eta=eta, quantile_value=quantile: np.sum(
                    np.cos(eta) * x[:, 0] + np.sin(eta) * x[:, 2] <= quantile_value
                )
                / len(x[:, 0])
            )
            # weights.append(1 / q / (1 - q))
            weights.append(1)
            logging.info(
                f"lambda x, eta=eta, quantile_value=quantile: np.mean(np.cos(eta) * x[:, 0] + np.sin(eta) * x[:, 2] <= quantile_value);q={q:00.2f};eta={eta:00.2f};quantile={quantile}; weight: "
                + str(weights[-1])
            )
            sen.append(weights[-1] / len(df))

    statistics.append(lambda x: np.sum(x[:, 1][x[:, 1] == True]) / len(x[:, 1]))
    weights.append(1)
    logging.info(
        f"lambda x: np.sum(x[:, 1][x[:, 1] == True]) / len(x[:, 1]); weight: "
        + str(weights[-1])
    )
    sen.append(weights[-1] / len(df))

    l_3 = 50
    for i in range(1, l_3):
        q = i / l_3
        quantile = np.quantile(df.values[:, 2], q)
        statistics.append(
            lambda x, quantile_value=quantile: np.sum(
                np.logical_and(x[:, 1] == True, x[:, 2] <= quantile_value)
            )
            / len(x[:, 2])
        )
        weights.append(1)
        logging.info(
            f"lambda x, quantile_value=quantile: np.mean(np.logical_and(x[:, 1] == True, x[:, 2] <= quantile_value));q={q:00.2f};quantile={quantile}; weight: "
            + str(weights[-1])
        )
        sen.append(weights[-1] / len(df))

    return statistics, weights, np.max(sen), stats


def proposal_distribution(
    df: pd.DataFrame, x: pd.DataFrame, sigma: list, stepsize: list = [1, 1]
):
    result = df.copy()
    i = np.random.choice(range(len(df)), replace=False, size=stepsize[0])
    columns = np.random.choice(df.columns, replace=False, size=stepsize[1])
    # columns = ["bmi"]
    for l in i:
        for column in columns:
            if column == "bmi":
                d = result[column].values[l] + np.random.normal(0, sigma[0])
                if d < min_0:
                    result[column].values[l] = max_0 - min_0 + d
                    while not (
                        result[column].values[l] < max_0
                        and result[column].values[l] > min_0
                    ):
                        result[column].values[l] += max_0 - min_0
                elif d > max_0:
                    result[column].values[l] = min_0 - max_0 + d
                    while not (
                        result[column].values[l] < max_0
                        and result[column].values[l] > min_0
                    ):
                        result[column].values[l] += min_0 - max_0
                else:
                    result[column].values[l] = d
            elif column == "smoker":
                result[column].values[l] = np.random.choice(
                    x[column].values[x[column].values != df[column].values[l]]
                )
            elif column == "charges":
                d = result[column].values[l] + np.random.normal(0, sigma[1])
                if d < min_2:
                    result[column].values[l] = max_2 - min_2 + d
                    while not (
                        result[column].values[l] < max_2
                        and result[column].values[l] > min_2
                    ):
                        result[column].values[l] += max_2 - min_2
                elif d > max_2:
                    result[column].values[l] = min_2 - max_2 + d
                    while not (
                        result[column].values[l] < max_2
                        and result[column].values[l] > min_2
                    ):
                        result[column].values[l] += min_2 - max_2
                else:
                    result[column].values[l] = d
    return result


def create_artificial_data(input_rows):
    rows_list = []
    for _ in range(input_rows):
        dict1 = {}
        p = 2000 * np.random.beta(a=2, b=2)
        k = np.random.randint(low=1, high=11)
        dict1.update(
            {"Einzelpreis": round(p, 2), "Anzahl": k, "Betrag": round(p * k, 2)}
        )
        rows_list.append(dict1)

    if not os.path.exists("data"):
        os.makedirs("data")

    df = pd.DataFrame(rows_list)
    df.to_csv(
        "data/001_Preis,Anzahl,Betrag.csv",
        index=False,
    )

    correlation_matrix_original_data = pd.DataFrame(np.corrcoef(np.transpose(df)))
    correlation_matrix_original_data.to_csv(
        "data/001_Preis,Anzahl,Betrag_correlation_matrix_original_data.csv",
        index=False,
    )

    return df


def load_data(filename: str):
    # df = artificial_data(rows)
    save = True
    # rows = 20
    database_path = "data/" + filename
    if rows == None:
        df = pd.read_csv(database_path, encoding="utf8")
    else:
        df = pd.read_csv(database_path, encoding="utf8").sample(n=rows)
    # df = data_preparation(df, filename=filename)
    if save and not rows == None:
        df.to_csv(
            "data/"
            + filename.split(".")[0]
            + f"_{rows}_rows."
            + filename.split(".")[1],
            index=False,
        )
    return df


def data_preparation(df: pd.DataFrame, filename: str):
    df = df.dropna()
    df = df.drop(df.columns[5], axis=1)
    df = df.drop(df.columns[3], axis=1)
    df = df.drop(df.columns[1], axis=1)
    df = df.drop(df.columns[0], axis=1)
    for i in range(len(df)):
        if df["smoker"][i] == "yes":
            df["smoker"][i] = 1
        else:
            df["smoker"][i] = 0
    df = df.astype(
        {
            "bmi": float,
            "smoker": bool,
            "charges": float,
        }
    )
    df.to_csv(
        "data/" + filename.split(".")[0] + "_without_nan." + filename.split(".")[1],
        index=False,
    )
    return df


def creating_differential_private_dataset(
    df: pd.DataFrame,
    data_name: str,
    alpha: float,
    k: int,
    iterations: int,
    stepsize: list,
    sigma: list,
):

    state = np.random.get_state()

    # state = (
    #     "MT19937",
    #     np.array(
    #         [
    #             2147483648,
    #             3795128051,
    #             4015480184,
    #             1426669335,
    #             3773292058,
    #             2261141328,
    #             2464151344,
    #             1286770076,
    #             246011732,
    #             636499926,
    #             1153222803,
    #             3104736112,
    #             3112774836,
    #             3996231751,
    #             261148543,
    #             1359901312,
    #             2230202579,
    #             3549135501,
    #             2186894070,
    #             708928415,
    #             3522057363,
    #             2088093091,
    #             1407488139,
    #             2819483875,
    #             895350048,
    #             1202600392,
    #             1927098738,
    #             3080483234,
    #             3338799295,
    #             386723002,
    #             3014356164,
    #             2514573999,
    #             3710573463,
    #             3019291056,
    #             1868108150,
    #             1713265792,
    #             2617140739,
    #             3078598761,
    #             3445931147,
    #             3878875804,
    #             2682893588,
    #             1093066368,
    #             3972019951,
    #             3759785703,
    #             2763809409,
    #             1893321519,
    #             1954756716,
    #             1621046417,
    #             2458111934,
    #             2513410889,
    #             60090171,
    #             1783196695,
    #             1327099748,
    #             147953543,
    #             336225637,
    #             765630668,
    #             1785794828,
    #             464220237,
    #             1386274063,
    #             2951183705,
    #             3310101489,
    #             1574346976,
    #             1339565239,
    #             2524890544,
    #             3880915183,
    #             2798911762,
    #             734504936,
    #             3510470179,
    #             1364645803,
    #             2756926661,
    #             3001511176,
    #             1232905042,
    #             3151837728,
    #             4157905650,
    #             4089042326,
    #             365397689,
    #             1504385174,
    #             1758033343,
    #             2827719268,
    #             3220785015,
    #             2133083929,
    #             2172276092,
    #             2984608586,
    #             3080303943,
    #             62878402,
    #             2919193897,
    #             3325590359,
    #             3729834942,
    #             2111156648,
    #             738309479,
    #             2692365781,
    #             3279775811,
    #             1554336536,
    #             1177146527,
    #             2387957562,
    #             75701453,
    #             1643204428,
    #             1420398063,
    #             3927224891,
    #             253017226,
    #             2201445066,
    #             1808504285,
    #             2934654212,
    #             3979356062,
    #             2729472077,
    #             3745036702,
    #             2343911322,
    #             3093728152,
    #             538839587,
    #             3959537399,
    #             1765322711,
    #             159388814,
    #             694517848,
    #             3971117364,
    #             2789955812,
    #             4176594863,
    #             1797342044,
    #             45159655,
    #             4123762340,
    #             2096393882,
    #             38773042,
    #             4249358254,
    #             1457323891,
    #             2291646249,
    #             2258048042,
    #             634080765,
    #             4223508178,
    #             691287546,
    #             3996217740,
    #             3845657598,
    #             3409177884,
    #             1032875406,
    #             4225501473,
    #             2603079857,
    #             2635908339,
    #             2261424612,
    #             1315357595,
    #             3882373966,
    #             2594265742,
    #             3517706709,
    #             871234256,
    #             289249611,
    #             1047366274,
    #             2513048644,
    #             3973827814,
    #             2132452380,
    #             1733897206,
    #             3547446447,
    #             4061268847,
    #             1793249588,
    #             4210314651,
    #             2422576779,
    #             260799478,
    #             322679645,
    #             347317020,
    #             4098224251,
    #             4187062828,
    #             1173878849,
    #             2493007863,
    #             385088445,
    #             3174352777,
    #             4168543280,
    #             3198831749,
    #             4280518589,
    #             1083173234,
    #             3978322998,
    #             492533940,
    #             3793523122,
    #             1546537683,
    #             3421538902,
    #             119228724,
    #             3560216778,
    #             1799149825,
    #             3416143464,
    #             3101095993,
    #             465011129,
    #             762362957,
    #             3914691027,
    #             1779315361,
    #             2839840026,
    #             2871312217,
    #             3913238633,
    #             2285432151,
    #             3311861256,
    #             228811486,
    #             3146856055,
    #             904738370,
    #             2645511441,
    #             476579541,
    #             3963157262,
    #             4034899728,
    #             639064025,
    #             2481369179,
    #             4131997583,
    #             2483442729,
    #             1427750919,
    #             35027179,
    #             1009655480,
    #             1872100474,
    #             840645173,
    #             4173892917,
    #             164038369,
    #             2525826210,
    #             4078602852,
    #             2392482727,
    #             3761039194,
    #             3476611543,
    #             660625902,
    #             2625105994,
    #             2824666714,
    #             3161987799,
    #             3823536952,
    #             3571304864,
    #             1187318803,
    #             4132996919,
    #             4127919589,
    #             1307309462,
    #             1000374470,
    #             266133107,
    #             1938825902,
    #             2198241467,
    #             1993726563,
    #             1318749173,
    #             1800363265,
    #             41246920,
    #             3962197868,
    #             1862116050,
    #             500427853,
    #             3900407174,
    #             4108180683,
    #             2528266589,
    #             3843945907,
    #             3157590728,
    #             2619649154,
    #             3328978499,
    #             192215690,
    #             4062604315,
    #             2053195415,
    #             3474945520,
    #             1105322735,
    #             191897165,
    #             923650187,
    #             3520701302,
    #             2970162263,
    #             3991105950,
    #             2197828438,
    #             1853632787,
    #             1739053001,
    #             2563084216,
    #             597761863,
    #             390138606,
    #             3346275018,
    #             1685379785,
    #             770437085,
    #             3216828115,
    #             3829915273,
    #             1144213390,
    #             432718092,
    #             3551938969,
    #             3343976783,
    #             1790407754,
    #             3530932195,
    #             1335771761,
    #             1654540208,
    #             2224012658,
    #             1671503239,
    #             420097317,
    #             1762703214,
    #             3557663024,
    #             1497639730,
    #             3098415169,
    #             1282624509,
    #             418794699,
    #             868814543,
    #             956627371,
    #             920214279,
    #             1389246387,
    #             1060252388,
    #             2078806151,
    #             2087622784,
    #             2585393764,
    #             2455795718,
    #             2150804381,
    #             722092340,
    #             3020970925,
    #             1699721977,
    #             3111222814,
    #             2434605651,
    #             3234276986,
    #             3795850246,
    #             4020799879,
    #             569212787,
    #             2979638372,
    #             203669301,
    #             2508704872,
    #             1556817253,
    #             2714528561,
    #             2026271584,
    #             3139270962,
    #             2181646583,
    #             3327516776,
    #             4204273006,
    #             2219946174,
    #             610409580,
    #             460514895,
    #             2880590132,
    #             3977208224,
    #             3402621073,
    #             453761910,
    #             1087093661,
    #             1828533650,
    #             491717665,
    #             138629950,
    #             1537444933,
    #             2323496402,
    #             2878332209,
    #             1417482944,
    #             3761715457,
    #             2286869281,
    #             2650581554,
    #             1901340393,
    #             3805114635,
    #             2594090673,
    #             188379172,
    #             397078871,
    #             2792751840,
    #             1083098977,
    #             1543770666,
    #             2694824290,
    #             2941875839,
    #             1290486468,
    #             2898697144,
    #             369046722,
    #             1983921974,
    #             747949691,
    #             270929064,
    #             2718661174,
    #             2936149764,
    #             3887660311,
    #             1091397587,
    #             3988880572,
    #             3858803414,
    #             336818556,
    #             3088726516,
    #             3223432284,
    #             2364049806,
    #             3079467555,
    #             589736984,
    #             67466168,
    #             4011229600,
    #             1564711980,
    #             4275700372,
    #             133744965,
    #             914326329,
    #             2881203092,
    #             1178571169,
    #             1034714193,
    #             326941034,
    #             2654359001,
    #             1790329880,
    #             3428105897,
    #             550236781,
    #             1946363948,
    #             3772184360,
    #             3797313643,
    #             2380067626,
    #             4064517292,
    #             3153312509,
    #             4280661528,
    #             546969884,
    #             3163380258,
    #             3683891405,
    #             3993626228,
    #             2263102642,
    #             2011571337,
    #             1955332834,
    #             2091220952,
    #             1010131667,
    #             1254742886,
    #             1751036896,
    #             1909425010,
    #             3243591011,
    #             3424616816,
    #             1036926208,
    #             4225352804,
    #             4028850917,
    #             1759667917,
    #             4178152727,
    #             2750918922,
    #             1556067113,
    #             3551253060,
    #             3751632051,
    #             3675469550,
    #             3609917196,
    #             3977921845,
    #             3330867959,
    #             277084606,
    #             254327646,
    #             3479147440,
    #             599213625,
    #             517250793,
    #             962301533,
    #             2169746940,
    #             179411311,
    #             3609693786,
    #             3519530513,
    #             2008492248,
    #             3569335889,
    #             986017464,
    #             2247526800,
    #             3732488662,
    #             315013880,
    #             1842460268,
    #             270698900,
    #             2574393885,
    #             1008201049,
    #             3465860137,
    #             1262336891,
    #             1091345187,
    #             263885604,
    #             2078274082,
    #             3223319817,
    #             1003386401,
    #             3923590771,
    #             2104691598,
    #             3254217809,
    #             1757961959,
    #             1912636158,
    #             1208130111,
    #             922754806,
    #             3888647975,
    #             880953996,
    #             744086884,
    #             1436115231,
    #             3763986678,
    #             3004769095,
    #             2928581363,
    #             2399760991,
    #             3442966616,
    #             1262146644,
    #             1305952938,
    #             4184932503,
    #             171614613,
    #             2992621281,
    #             4275487851,
    #             888567744,
    #             2852777318,
    #             3255408583,
    #             3043849936,
    #             3033069838,
    #             4259141106,
    #             3139305984,
    #             4079397339,
    #             726226399,
    #             3649086063,
    #             1686767780,
    #             2015129057,
    #             949240754,
    #             2509494204,
    #             3860986719,
    #             2872305166,
    #             3364350961,
    #             1964090366,
    #             2754463395,
    #             3454534101,
    #             2553784881,
    #             2178666356,
    #             4110580188,
    #             4116449313,
    #             3541050822,
    #             3557264744,
    #             723416304,
    #             1104763300,
    #             1171219113,
    #             1128725225,
    #             2243784835,
    #             2661596441,
    #             158454215,
    #             1502358755,
    #             4226802081,
    #             1161643904,
    #             8951758,
    #             1949499420,
    #             2379405388,
    #             721788664,
    #             537830424,
    #             3964480418,
    #             2378889572,
    #             3112642426,
    #             2681459573,
    #             725744494,
    #             2504851635,
    #             1623858935,
    #             2513800483,
    #             1006262549,
    #             1430835782,
    #             2394028940,
    #             2004741371,
    #             1735081412,
    #             3154080417,
    #             3481219842,
    #             1903308068,
    #             1956852684,
    #             3783849446,
    #             354984533,
    #             1059416003,
    #             1520131892,
    #             3338946234,
    #             1547748093,
    #             2987744291,
    #             3326003025,
    #             2392166491,
    #             3113243181,
    #             3222516949,
    #             1497701226,
    #             219348748,
    #             3899889294,
    #             4148969178,
    #             2293267807,
    #             3898171767,
    #             1586136951,
    #             1629944576,
    #             1546666015,
    #             752332457,
    #             2778034344,
    #             1816011336,
    #             1265638303,
    #             568052309,
    #             2141703066,
    #             1773695166,
    #             2991200997,
    #             2949463571,
    #             2438992925,
    #             4292558349,
    #             3200249296,
    #             2845696371,
    #             2725503502,
    #             4176215693,
    #             947907536,
    #             2765400810,
    #             3201069425,
    #             3424702358,
    #             3289361248,
    #             362857198,
    #             1812015804,
    #             3931125782,
    #             139871865,
    #             390895058,
    #             2999832368,
    #             3019308256,
    #             91620505,
    #             2429848954,
    #             3948069295,
    #             2724824527,
    #             1042502668,
    #             584993824,
    #             3314779054,
    #             1015611262,
    #             3071503356,
    #             596197255,
    #             968647809,
    #             3409862072,
    #             3930496745,
    #             592538895,
    #             3521343902,
    #             426360383,
    #             3286811271,
    #             4250894385,
    #             2562728893,
    #             640931285,
    #             1677687888,
    #             2443676441,
    #             847320124,
    #             1301694624,
    #             544987924,
    #             313919864,
    #             3414359103,
    #             2108838516,
    #             3389576743,
    #             1479001913,
    #             1415926424,
    #             2213708644,
    #             281640742,
    #             296199049,
    #             2146221585,
    #             1965195268,
    #             1591732464,
    #             1850258742,
    #             4076188616,
    #             1327049327,
    #             2088416917,
    #             1946645641,
    #             3798325848,
    #             2771333417,
    #             3051779913,
    #             2256391331,
    #             3455401096,
    #             2594398837,
    #             806518141,
    #             1983157210,
    #             1694410883,
    #             2984677504,
    #             3903440134,
    #             626792034,
    #             52663454,
    #             1832581004,
    #             3066683302,
    #             2439147820,
    #             1361215391,
    #             2618572114,
    #             4103347387,
    #             1314567620,
    #             1697400017,
    #             1784704118,
    #             2647254566,
    #             3477822949,
    #             4209420315,
    #             3948545532,
    #             3425208327,
    #             2780175944,
    #             46967025,
    #             1551369870,
    #             484002358,
    #             3007944263,
    #         ],
    #         dtype=np.uint32,
    #     ),
    #     623,
    #     0,
    #     0.0,
    # )

    # np.random.set_state(state)

    settings = (
        "alpha="
        + str(alpha)
        + ";k="
        + str(k)
        + ";iterations="
        + str(iterations)
        + ";stepsize="
        + str(stepsize)
        + ";sigma="
        + str(sigma)
    )

    if not os.path.exists("results/" + data_name + "/" + settings):
        os.makedirs("results/" + data_name + "/" + settings)

    count = 1

    while True:
        if not os.path.exists(
            "results/" + data_name + "/" + settings + "/" + str(count)
        ):
            os.makedirs("results/" + data_name + "/" + settings + "/" + str(count))
            break
        count += 1

    logging.basicConfig(
        filename="results/"
        + data_name
        + "/"
        + settings
        + "/"
        + str(count)
        + "/seed_statistics_weights.log",
        level=logging.INFO,
        filemode="w",
        force=True,
    )

    logging.info(state)

    statistics, weights, sen, stats = weights_and_statistics_and_sensitivity_and_stats(
        df
    )

    e = Exponential_Mechanism(statistics=statistics, x=df, weights=weights, alpha=alpha)

    (
        counter_2,
        min,
        synthetic_dataframe,
        counter,
        l,
        result_diff,
        result_x,
        result_z,
        result_query,
        starting_point,
    ) = e.metropol_haste(
        k=k,
        proposal_distribution=proposal_distribution,
        stats=stats,
        sigma=sigma,
        iterations=iterations,
        sensitivity=sen,
        stepsize=stepsize,
    )

    logging.info("Accepted: " + str(counter) + " / " + str(l))
    logging.info("Accepted with AR < 1: " + str(counter_2) + " / " + str(counter))
    logging.info("Smallest Accepted_Ratio: " + str(min))

    starting_point.to_csv(
        "results/"
        + data_name
        + "/"
        + settings
        + "/"
        + str(count)
        + "/starting_point.csv",
        index=False,
    )

    pd.DataFrame(np.array(result_query)).to_csv(
        "results/"
        + data_name
        + "/"
        + settings
        + "/"
        + str(count)
        + "/result_query.csv",
        index=False,
    )

    pd.DataFrame(result_z).to_csv(
        "results/" + data_name + "/" + settings + "/" + str(count) + "/result_z.csv",
        index=False,
    )

    pd.DataFrame(result_x).to_csv(
        "results/" + data_name + "/" + settings + "/" + str(count) + "/result_x.csv",
        index=False,
    )

    pd.DataFrame(result_diff).to_csv(
        "results/" + data_name + "/" + settings + "/" + str(count) + "/result_diff.csv",
        index=False,
    )

    synthetic_dataframe.to_csv(
        "results/"
        + data_name
        + "/"
        + settings
        + "/"
        + str(count)
        + "/synthetic_data.csv",
        index=False,
    )

    return None


def main1():

    filename = "insurance_without_nan.csv"
    # original_database = load_data(filename=filename)
    original_database = pd.read_csv("data/" + filename, encoding="utf8")

    original_database.info()

    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists("results/" + filename):
        os.makedirs("results/" + filename)

    alphas = [0.2]
    ks = [20]
    stepsizes = [[1, 1]]
    sigma = [1, 500]
    iterations = 500_000

    for k in ks:
        for alpha in alphas:
            for stepsize in stepsizes:
                for _ in range(1):
                    creating_differential_private_dataset(
                        df=original_database,
                        data_name=filename,
                        alpha=alpha,
                        k=k,
                        iterations=iterations,
                        stepsize=stepsize,
                        sigma=sigma,
                    )
    return None


def help_z(s: list):
    result = []
    f1 = s[0].split(" ")
    f1 = [x for x in f1 if x != ""]
    f2 = s[1].split(" ")
    f2 = [x for x in f2 if x != ""]
    f3 = s[2].split(" ")
    f3 = [x for x in f3 if x != ""]

    result.append(float(f1[0]))
    result.append(float(f1[1]))
    result.append(float(f1[2]))
    result.append(float(f2[0]))
    result.append(float(f2[1]))
    result.append(float(f2[2]))
    result.append(float(f3[0]))
    result.append(float(f3[1]))
    result.append(float(f3[2]))
    result = np.array(result).reshape((3, 3))
    return result


def help_x(s: list):
    s = s.replace("'", "").replace("\\n", "")
    return np.array(s.split(" ")).astype("float").reshape((3, 3))


def analyse_original_data():
    filename = "insurance_without_nan_100_rows.csv"
    df = pd.read_csv("data/" + filename, encoding="utf8")
    df.info()

    cor = df.corr()

    # Histogramm BMI neben Kosten
    cm = 1/2.54
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16 / 2, 9 / 2))
    ax[0].hist(df["bmi"], bins=np.multiply(2, np.array(range(7, 26))))
    # ax[0].set_title("Histogramm vom Attribut BMI")
    ax[0].set_xlabel("BMI")
    ax[0].set_ylabel("Anzahl")
    # ax[1].hist(df["smoker"].astype("float"))
    # ax[1].set_title("Histogramm vom Attribut Raucher")
    ax[1].hist(df["charges"], bins=np.multiply(2000, np.array(range(1, 28))))
    # ax[1].set_title("Histogramm vom Attribut Kosten")
    ax[1].set_xlabel("Kosten")
    fig.tight_layout()
    # plt.show()
    plt.savefig("pictures/histogramm_bmi_charges.png", dpi=600)

    # Histogramm Kosten auf Untergruppen Raucherstatus
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16 / 2, 9 / 2))
    ax[0].hist(
        df["charges"][df["smoker"] == False],
        bins=np.multiply(2000, np.array(range(1, 28))),
    )
    # ax[0].set_title("Histogramm vom Attribut Kosten")
    ax[0].set_xlabel("Kosten für Nichtraucher")
    ax[0].set_ylabel("Anzahl")
    ax[1].hist(
        df["charges"][df["smoker"] == True],
        bins=np.multiply(2000, np.array(range(1, 28))),
    )
    # ax[1].set_title("Histogramm vom Attribut Kosten")
    ax[1].set_xlabel("Kosten für Raucher")
    fig.tight_layout()
    # plt.show()
    plt.savefig("pictures/histogramm_charges_subgroup_smoker.png", dpi=600)

    return None


def analyse_exponential_mechanism():
    global cnames
    filepath1 = "results/insurance_without_nan_100_rows.csv/alpha=0.1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"
    filepath2 = "results/insurance_without_nan_100_rows.csv/alpha=1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"
    filepath3 = "results/insurance_without_nan_100_rows.csv/alpha=10;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"
    # filepath4 = "results/insurance_without_nan_100_rows.csv/alpha=100;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"

    synthetic_df, result_x, result_z = [], [], []
    for path in [filepath1, filepath2, filepath3]:
        synthetic_df.append(pd.read_csv(path + "synthetic_data.csv"))
        result_x.append(np.transpose(pd.read_csv(path + "result_x.csv")))
        result_z.append(pd.read_csv(path + "result_z.csv"))

    # Entwicklung wärend MCMC
    stepsize = 500
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16 / 2, 9))

    for k, i in enumerate([4, 3, 7]):  # 8 ist Kovarianzmatrix
        ax[k].axhline(y=round(float(result_x[0][i]), 2), color=cnames.get("red"))
        for l, color in enumerate(
            [cnames.get("orange"), cnames.get("blue"), cnames.get("darkcyan")]
        ):
            ax[k].plot(
                [
                    x + stepsize - 1
                    for x in range(0, len(result_z[l].values[:, i]) - 1, stepsize)
                ],
                result_z[l].values[1::stepsize, i],
                color=color,
            )
    ax[0].set_title("Schiefe der Kosten", fontsize=10)
    ax[1].set_title("Mittelwert der Kosten", fontsize=10)
    ax[2].set_title("Mittelwert der Kosten für Nichtraucher", fontsize=10)
    ax[2].set_xlabel("Iterationen")

    for i in [3, 7]:
        print("Originalwert: " + str(round(float(result_x[0][i]), 2)))
        print(
            "Alpha = 0.1: Mittelwert: "
            + str(round(result_z[0].values[50_000::1, i].mean(), 2))
            + " Standardabweichung: "
            + str(round(result_z[0].values[50_000::1, i].std(), 2))
        )
        print(
            "Alpha = 1: Mittelwert: "
            + str(round(result_z[1].values[50_000::1, i].mean(), 2))
            + " Standardabweichung: "
            + str(round(result_z[1].values[50_000::1, i].std(), 2))
        )
        print(
            "Alpha = 10: Mittelwert: "
            + str(round(result_z[2].values[50_000::1, i].mean(), 2))
            + " Standardabweichung: "
            + str(round(result_z[2].values[50_000::1, i].std(), 2))
        )

    orange = mpatches.Patch(color=cnames.get("orange"), label=r"$\alpha = 0{,}1$")
    green = mpatches.Patch(color=cnames.get("blue"), label=r"$\alpha = 1$")
    darkcyan = mpatches.Patch(color=cnames.get("darkcyan"), label=r"$\alpha = 10$")
    red = mpatches.Patch(color=cnames.get("red"), label="Wert der Originaldatenbank")

    fig.legend(
        handles=[orange, green, darkcyan, red],
        loc="lower center",
        fancybox=True,
        # shadow=True,
        ncol=4,
    )
    # plt.subplots_adjust(bottom=0.2)
    # plt.show()
    plt.savefig("pictures/sen_alpha.png", dpi=600)

    filepath1 = "results/insurance_without_nan_100_rows.csv/alpha=0.1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"
    filepath2 = "results/insurance_without_nan.csv/alpha=0.1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"
    filepath3 = "results/insurance_without_nan.csv/alpha=1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"

    synthetic_df, result_x, result_z = [], [], []
    for path in [filepath1, filepath2, filepath3]:
        synthetic_df.append(pd.read_csv(path + "synthetic_data.csv"))
        result_x.append(np.transpose(pd.read_csv(path + "result_x.csv")))
        result_z.append(pd.read_csv(path + "result_z.csv"))

    stepsize = 1000
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16 / 2, 9))

    for k, i in enumerate([4, 3, 8]):  # 8 ist Kovarianzmatrix
        if not i == 8:
            ax[k].axhline(
                y=round(float(result_x[0][i]), 2), color=cnames.get("red"), alpha=0.2
            )
            ax[k].axhline(y=round(float(result_x[1][i]), 2), color=cnames.get("red"))
            for l, color in enumerate(
                [cnames.get("orange"), cnames.get("blue"), cnames.get("darkcyan")]
            ):
                ax[k].plot(
                    [
                        x + stepsize - 1
                        for x in range(0, len(result_z[l].values[:, i]) - 1, stepsize)
                    ],
                    result_z[l].values[1::stepsize, i],
                    color=color,
                )
        else:
            x = (
                str(result_x[0][i].to_list())
                .replace("[", "")
                .replace("]", "")
                .split("\n")[0]
            )
            y = help_x(x)
            ax[k].axhline(
                y=round(
                    float(y[1, 2]) / np.sqrt(float(y[1, 1])) / np.sqrt(float(y[2, 2])),
                    2,
                ),
                color=cnames.get("red"),
                alpha=0.2,
            )
            x = (
                str(result_x[1][i].to_list())
                .replace("[", "")
                .replace("]", "")
                .split("\n")[0]
            )
            y = help_x(x)
            z = float(y[1, 2]) / np.sqrt(float(y[1, 1])) / np.sqrt(float(y[2, 2]))
            ax[k].axhline(
                y=round(z, 2),
                color=cnames.get("red"),
            )
            for l, color in enumerate(
                [cnames.get("orange"), cnames.get("blue"), cnames.get("darkcyan")]
            ):
                y = np.array(
                    [
                        help_z(
                            result_z[l]["8"]
                            .to_list()[o]
                            .replace("[", "")
                            .replace("]", "")
                            .split("\n")
                        )
                        for o in range(len(result_z[l].values[:, i]) - 1)
                    ]
                )
                v = (
                    y[1::stepsize, 1, 2]
                    / np.sqrt(y[1::stepsize, 1, 1])
                    / np.sqrt(y[1::stepsize, 2, 2])
                )
                ax[k].plot(
                    [
                        x + stepsize - 1
                        for x in range(0, len(result_z[l].values[:, i]) - 1, stepsize)
                    ],
                    v,
                    color=color,
                )
    ax[0].set_title("Schiefe der Kosten", fontsize=10)
    ax[1].set_title("Mittelwert der Kosten", fontsize=10)
    ax[2].set_title("Korrelation zwischen Raucher und Kosten", fontsize=10)
    ax[2].set_xlabel("Iterationen")

    orange = mpatches.Patch(
        color=cnames.get("orange"), label=r"$\alpha = 0{,}1$ für $n = 100$"
    )
    green = mpatches.Patch(
        color=cnames.get("blue"), label=r"$\alpha = 0{,}1$ für $n = 1338$"
    )
    darkcyan = mpatches.Patch(
        color=cnames.get("darkcyan"), label=r"$\alpha = 1$ für $n = 1338$"
    )
    red_trans = mpatches.Patch(
        color=cnames.get("red"),
        alpha=0.2,
        label=r"Wert der Originaldatenbank mit $n = 100$",
    )
    red = mpatches.Patch(
        color=cnames.get("red"), label=r"Wert der Originaldatenbank mit $n = 1338$"
    )

    fig.legend(
        handles=[orange, green, darkcyan, red, red_trans],
        loc="lower center",
        fancybox=True,
        # shadow=True,
        ncol=2,
    )
    plt.subplots_adjust(bottom=0.15)
    # plt.show()
    plt.savefig("pictures/sen_alpha_n.png", dpi=600)

    filepath = "results/insurance_without_nan.csv/alpha=1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/2/"

    synthetic_df, result_x, result_z = [], [], []
    for path in [filepath]:
        synthetic_df.append(pd.read_csv(path + "synthetic_data.csv"))
        result_x.append(np.transpose(pd.read_csv(path + "result_x.csv")))
        result_z.append(pd.read_csv(path + "result_z.csv"))

    x_cov = help_x(
        str(result_x[0][8].to_list()).replace("[", "").replace("]", "").split("\n")[0]
    )
    x_corr = []
    for i in range(3):
        for k in range(3):
            x_corr.append(x_cov[i, k] / np.sqrt(x_cov[i, i]) / np.sqrt(x_cov[k, k]))

    x_corr = np.array(x_corr).reshape((3, 3))

    z_cov = help_z(
        result_z[0]["8"].to_list()[-1].replace("[", "").replace("]", "").split("\n")
    )
    z_corr = []
    for i in range(3):
        for k in range(3):
            z_corr.append(z_cov[i, k] / np.sqrt(z_cov[i, i]) / np.sqrt(z_cov[k, k]))

    z_corr = np.array(z_corr).reshape((3, 3))

    filepath1 = "results/insurance_without_nan.csv/alpha=0.1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/2/"
    filepath2 = "results/insurance_without_nan.csv/alpha=0.2;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/1/"
    filepath3 = "results/insurance_without_nan.csv/alpha=1;k=20;iterations=100000;stepsize=[1, 1];sigma=[1, 500]/3/"

    result_query, result_x, result_z = [], [], []
    for path in [filepath1, filepath2, filepath3]:
        result_query.append(pd.read_csv(path + "result_query.csv"))
        result_x.append(np.transpose(pd.read_csv(path + "result_x.csv")))
        result_z.append(pd.read_csv(path + "result_z.csv"))

    # Entwicklung wärend MCMC
    stepsize = 500
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16 / 2, 9 / 2))

    x = [
        x + stepsize - 1
        for x in range(0, len(result_query[0].values[:, 0]) - 1, stepsize)
    ]
    color = [cnames.get("orange"), cnames.get("blue"), cnames.get("darkcyan")]
    k = 3  # 8 Kovarianzmatrix

    ax[1].axhline(y=round(float(result_x[0][k]), 2), color=cnames.get("red"), alpha=1)
    for i in range(len(result_query)):
        ax[0].plot(
            x,
            result_query[i].values[1::stepsize, 0],
            color=color[i],
        )
        ax[1].plot(
            x,
            result_z[i].values[1::stepsize, k],
            color=color[i],
        )

    ax[0].set_title("Bewertungsfunktion", fontsize=10)
    ax[1].set_title("Mittelwert der Kosten", fontsize=10)
    ax[1].set_xlabel("Iterationen")

    for i in [3]:
        print("Originalwert: " + str(round(float(result_x[0][i]), 2)))
        print(
            "Alpha = 0.1: Mittelwert: "
            + str(round(result_z[0].values[50_000::1, i].mean(), 2))
            + " Standardabweichung: "
            + str(round(result_z[0].values[50_000::1, i].std(), 2))
        )
        print(
            "Alpha = 0.2: Mittelwert: "
            + str(round(result_z[1].values[50_000::1, i].mean(), 2))
            + " Standardabweichung: "
            + str(round(result_z[1].values[50_000::1, i].std(), 2))
        )
        print(
            "Alpha = 1: Mittelwert: "
            + str(round(result_z[2].values[50_000::1, i].mean(), 2))
            + " Standardabweichung: "
            + str(round(result_z[2].values[50_000::1, i].std(), 2))
        )

    orange = mpatches.Patch(color=cnames.get("orange"), label=r"$\alpha = 0{,}1$")
    green = mpatches.Patch(color=cnames.get("blue"), label=r"$\alpha = 0{,}2$")
    darkcyan = mpatches.Patch(color=cnames.get("darkcyan"), label=r"$\alpha = 1$")
    red = mpatches.Patch(
        color=cnames.get("red"), label=r"Wert der Originaldatenbank mit $n = 1338$"
    )

    fig.legend(
        handles=[orange, green, darkcyan, red],
        loc="lower center",
        fancybox=True,
        # shadow=True,
        ncol=4,
    )
    plt.subplots_adjust(bottom=0.2)
    # plt.show()
    plt.savefig("pictures/query_function_iteration.png", dpi=600)

    filepath1 = "results/insurance_without_nan.csv/alpha=0.1;k=20;iterations=500000;stepsize=[1, 1];sigma=[1, 500]/1/"
    filepath2 = "results/insurance_without_nan.csv/alpha=0.2;k=20;iterations=500000;stepsize=[1, 1];sigma=[1, 500]/1/"

    result_query, result_x, result_z = [], [], []
    for path in [filepath1, filepath2]:
        result_query.append(pd.read_csv(path + "result_query.csv"))
        result_x.append(np.transpose(pd.read_csv(path + "result_x.csv")))
        result_z.append(pd.read_csv(path + "result_z.csv"))

    # Entwicklung wärend MCMC
    stepsize = 500
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16 / 2, 9 / 2))

    x = [
        x + stepsize - 1
        for x in range(0, len(result_query[0].values[:, 0]) - 1, stepsize)
    ]
    color = [cnames.get("orange"), cnames.get("blue")]
    k = 3  # 8 Kovarianzmatrix

    ax[1].axhline(y=round(float(result_x[0][k]), 2), color=cnames.get("red"), alpha=1)
    for i in range(len(result_query)):
        ax[0].plot(
            x,
            result_query[i].values[1::stepsize, 0],
            color=color[i],
        )
        ax[1].plot(
            x,
            result_z[i].values[1::stepsize, k],
            color=color[i],
        )

    ax[0].set_title("Bewertungsfunktion", fontsize=10)
    ax[1].set_title("Mittelwert der Kosten", fontsize=10)
    ax[1].set_xlabel("Iterationen")

    orange = mpatches.Patch(color=cnames.get("orange"), label=r"$\alpha = 0{,}1$")
    green = mpatches.Patch(color=cnames.get("blue"), label=r"$\alpha = 0{,}2$")
    red = mpatches.Patch(
        color=cnames.get("red"), label=r"Wert der Originaldatenbank mit $n = 1338$"
    )

    fig.legend(
        handles=[orange, green, red],
        loc="lower center",
        fancybox=True,
        # shadow=True,
        ncol=4,
    )
    plt.subplots_adjust(bottom=0.2)
    # plt.show()
    plt.savefig("pictures/query_function_iteration_500000.png", dpi=600)

    return None


def main2():
    # analyse_original_data()
    analyse_exponential_mechanism()


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    # main1()
    main2()


