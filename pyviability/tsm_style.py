"""
Provide the colors and plotting styles used for TSM framework.
"""


from __future__ import print_function, division

import matplotlib as mpl
import os.path
import xml.etree.ElementTree as ET


colorFileName = "colors.soc"
"""relative path to the color file"""
colorpath = os.path.join(os.path.split(__file__)[0], colorFileName)
"""absolute path to the color file"""

tree = ET.parse(colorpath)
root = tree.getroot()

tsm_colors = {}
"""provide all colors from the TSM framework"""
for child in root.getchildren():
    tsm_colors[child.attrib['{TSM}name']] = child.attrib['{TSM}color']

del tree, root, child

color_translate_dict = {
    "Default Flow" : "cDefault",
    "management" :   "cMod",

    ## 		"TSM Upstream (borders)" : "
    "Shelters (upstream)" : "cShelter",
    "Glades (upstream)" : "cGlade",
    "Lakes (upstream)" : "cLake",
    "Sunny Upstream (remaining)" : "cSunnyUp",
    "Dark Upstream" : "cDarkUp",

    ## 		"TSM Downstream (borders)"
    "Backwaters (downstream)" : "cBackwaters",
    "Sunny Downstream (remaining)" : "cSunnyDown",
    "Dark Downstream" : "cDarkDown",

    ##   "TSM Eddies (border)"
    "Sunny Eddies" : "cSunnyEddie",
    "Dark Eddies" : "cDarkEddie",

    ##   "TSM Abysses (border)"
    "Sunny Abysses" : "cSunnyAbyss",
    "Dark Abysses" : "cDarkAbyss",

    "Trenches" : "cTrench",

    "Boundary" : "cBound",
    "Area Boundary" : "cAreaBound",
}
"""translating between the long keys and the shorter version"""

# check if all necessary colors are defined
assert set(tsm_colors.keys()).issuperset(color_translate_dict.keys()), "some colors are not defined (color file is %s)" % colorpath

# save colors with short keys, too
for colTop, colC in sorted(color_translate_dict.items()):
    tsm_colors[colC] = tsm_colors[colTop]

del color_translate_dict


styleDefault = {
    "linewidth": 3,
    "color":tsm_colors["cDefault"],
    "arrowsize": 1.5,
}
"""plotting kwargs for the default flow"""
styleMod1 = {
    "linewidth": 1,
    "color":tsm_colors["cMod"],
    "linestyle": "dotted",
    "arrowsize": 1.5,
}
"""plotting kwargs for the management flow, version 1"""
styleMod2 = {
    "linewidth": 1,
    "color":tsm_colors["cMod"],
    "linestyle": "--",
    "arrowsize": 1.5,
}
"""plotting kwargs for the management flow, version 2"""


stylePatch = {
    "linewidth": 2,
    "closed": True,
    "fill": True,
    "edgecolor": tsm_colors["cAreaBound"],
    "alpha": 0.5,
}
"""kwargs for the patch creation for a 2D-plot of a TSM partition"""\
    """ (color not given because that is region dependent)"""

stylePoint = dict(markersize=10)
"""basic style of a point used in TSM plots"""

# adjust global plotting style
mpl.rcParams["axes.labelsize"] = 24
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["font.size"] = 30

if __name__ == "__main__":
    import json
    print("allcolors =", json.dumps(tsm_colors, indent=4, sort_keys=True))


