from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#region Enums
class CapShape(Enum):
        b = 0
        c = 1
        x = 2
        f = 3
        s = 4
        p = 5
        o = 6

class Surface(Enum):
        i = 1
        g = 2
        y = 3
        s = 4
        h = 5
        l = 6
        k = 7
        t = 8
        w = 9
        e = 10
        d = 11
        f = 12

class Color(Enum):
        n = 1
        b = 2
        g = 3
        r = 4
        p = 5
        u = 6
        e = 7
        w = 8
        y = 9
        l = 10
        o = 11
        k = 12
        f = 13

class GillAttach(Enum):
        empty = 0
        a = 1
        x = 2
        d = 3
        e = 4 
        s = 5
        p = 6
        f = 7

class GillSpacing(Enum):
        empty = 0
        c = 1
        d = 2
        f = 3

class StemRoot(Enum):
        empty = 0
        b = 1
        s = 2
        c = 3
        u = 4
        e = 5
        z = 6
        r = 7
        f = 8

class VeilType(Enum):
        empty = 0
        p = 1
        u = 2

class RingType(Enum):
        empty = 0
        c = 1
        e = 2
        r = 3
        g = 4
        l = 5
        p = 6
        s = 7
        z = 8
        y = 9
        m = 10
        f = 11

class Habitat(Enum):
        g = 0
        l = 1
        m = 2
        p = 3
        h = 4
        u = 5
        w = 6
        d = 7

class Season(Enum):
        s = 0
        u = 1
        a = 2
        w = 3
#endregion

def prepareData(data):
    data['class'] = [int(i == 'e') for i  in data['class']]

    for i in CapShape._member_names_:
        data['cap-shape-' + i] = np.where(data['cap-shape'] == i, 1, 0)
    data.pop('cap-shape')

    for i in Surface._member_names_:
        data['cap-surface-' + i] = np.where(data['cap-surface'] == i, 1, 0)
    data.pop('cap-surface')

    for i in Color._member_names_:
        data['cap-color-' + i] = np.where(data['cap-color'] == i, 1, 0)
    data.pop('cap-color')
    
    data['does-bruise-or-bleed'] = [int(i == 't') for i  in data['does-bruise-or-bleed']]

    for i in GillAttach._member_names_:
        data['gill-attachment-' + i] = np.where(data['gill-attachment'] == i, 1, 0)
    data.pop('gill-attachment')

    for i in GillSpacing._member_names_:
        data['gill-spacing-' + i] = np.where(data['gill-spacing'] == i, 1, 0)
    data.pop('gill-spacing')

    for i in Color._member_names_:
        data['gill-color-' + i] = np.where(data['gill-color'] == i, 1, 0)
    data.pop('gill-color')

    for i in StemRoot._member_names_:
        data['stem-root-' + i] = np.where(data['stem-root'] == i, 1, 0)
    data.pop('stem-root')

    for i in Surface._member_names_:
        data['stem-surface-' + i] = np.where(data['stem-surface'] == i, 1, 0)
    data.pop('stem-surface')

    for i in Color._member_names_:
        data['stem-color-' + i] = np.where(data['stem-color'] == i, 1, 0)
    data.pop('stem-color')
    
    for i in VeilType._member_names_:
        data['veil-type-' + i] = np.where(data['veil-type'] == i, 1, 0)
    data.pop('veil-type')

    for i in Color._member_names_:
        data['veil-color-' + i] = np.where(data['veil-color'] == i, 1, 0)
    data.pop('veil-color')
    
    data['has-ring'] = [int(i == 't') for i  in data['has-ring']]

    for i in RingType._member_names_:
        data['ring-type-' + i] = np.where(data['ring-type'] == i, 1, 0)
    data.pop('ring-type')

    for i in Color._member_names_:
        data['spore-print-color-' + i] = np.where(data['spore-print-color'] == i, 1, 0)
    data.pop('spore-print-color')

    for i in Habitat._member_names_:
        data['habitat-' + i] = np.where(data['habitat'] == i, 1, 0)
    data.pop('habitat')

    for i in Season._member_names_:
        data['season-' + i] = np.where(data['season'] == i, 1, 0)
    data.pop('season')

    print("\nINITIAL VARIABLES: " + str(len(list(data.columns))))
    print("\nCOLUMNS WITH ALL 0 VALUES:\n")

    for i in (list(data.columns)):
        if(data[i] == 0).all():
            print(i)
            data.pop(i)

    print("\nTOTAL VARIABLES: " + str(len(list(data.columns))))
    print()

def createGraphMetricalValue(data, edibles, poisonous, tag, title, xLabel, yLabel, fileName):
    plt.plot(np.arange(len(data[tag][edibles])), np.sort(data[tag][edibles])[::-1], label='Edible')
    plt.plot(np.arange(len(data[tag][poisonous])), np.sort(data[tag][poisonous])[::-1], c='#ff7f0e', label='Poisonous')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig('../Results/Analysis/' + fileName, bbox_inches='tight')
    plt.close()

def createGraphBinaryValue(data, edibles, poisonous, tag, labels, width, title, fileName):
    x = np.arange(len(labels))  # the label locations

    edib = [len(np.where(data[tag][edibles])[0]), len(np.where(data[tag][edibles] == 0)[0])]
    pois = [len(np.where(data[tag][poisonous])[0]), len(np.where(data[tag][poisonous] == 0)[0])]

    fig, ax = plt.subplots()
    ax.bar(x - width/2, edib, width, label='Edible')
    ax.bar(x + width/2, pois, width, label='Poisonous')
    plt.ylabel('Mushrooms')
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig('../Results/Analysis/' + fileName, bbox_inches='tight')
    plt.close()

def createGraphEnumValue(data, edibles, poisonous, tags, labels, width, title, fileName):
    x = np.arange(len(labels))  # the label locations

    edib = []
    pois = []

    for tag in tags:
        edib.append(len(np.where(data[tag][edibles])[0]))
        pois.append(len(np.where(data[tag][poisonous])[0]))

    fig, ax = plt.subplots(figsize=(16,9))
    ax.bar(x - width/2, edib, width, label='Edible')
    ax.bar(x + width/2, pois, width, label='Poisonous')
    plt.ylabel('Mushrooms')
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig('../Results/Analysis/' + fileName, bbox_inches='tight')
    plt.close()

def analyzeData(data):
    print(data.head())

    print(data.describe())

    edibles = np.where(data['class'])[0]
    poisonous = np.where(data['class'] == 0)[0]

    numEdibles = len(edibles)
    numPoisonous = len(poisonous)

    classMush = [numEdibles, numPoisonous]

    # 00 number of edibles and poisonous
    plt.bar(np.arange(2), classMush)[1].set_color('#ff7f0e')
    plt.xticks(np.arange(2), ['Edibles: ' + str(len(edibles)), 'Poisonus: ' + str(len(poisonous))])
    plt.ylabel('Quantity')
    plt.title('Number of poisonous and edibles mushrooms')
    plt.savefig('../Results/Analysis/00EdiblesNumb.png', bbox_inches='tight')
    plt.close()

    # 01 cap diameter
    createGraphMetricalValue(data, edibles, poisonous, 'cap-diameter',
     'Cap diameter of poisonous and edible mushrooms', 'Mushroom', 'Diameter', '01CapDiameter.png')

    # 02 cap shape
    createGraphEnumValue(data, edibles, poisonous,
    ['cap-shape-b', 'cap-shape-c', 'cap-shape-x', 'cap-shape-f', 'cap-shape-s', 'cap-shape-p', 'cap-shape-o'],
    ['Bell', 'Conical', 'Convex', 'Flat', 'Sunken', 'Spherical', 'Other'], 0.35,
    'Cap shape by class', '02CapShape.png')

    # 03 cap surface
    createGraphEnumValue(data, edibles, poisonous,
    ['cap-surface-i', 'cap-surface-g', 'cap-surface-y', 'cap-surface-s', 'cap-surface-h', 'cap-surface-l',
     'cap-surface-k', 'cap-surface-t', 'cap-surface-w', 'cap-surface-e', 'cap-surface-d'],
    ['Fibrous', 'Grooves', 'Scaly', 'Smooth', 'Shiny', 'Leatherly',
     'Silky', 'Sticky', 'Wrinkled', 'Fleshy', 'd(Unknown)'], 0.35,
    'Cap surface by class', '03CapSurface.png')

    # 04 cap color
    createGraphEnumValue(data, edibles, poisonous,
    ['cap-color-n', 'cap-color-b', 'cap-color-g', 'cap-color-r', 'cap-color-p', 'cap-color-u',
    'cap-color-e', 'cap-color-w', 'cap-color-y', 'cap-color-l', 'cap-color-o', 'cap-color-k'],
    ['Brown', 'Buff', 'Gray', 'Green', 'Pink', 'Purple',
    'Red', 'White', 'Yellow', 'Blue', 'Orange', 'Black'], 0.35,
    'Cap color by class', '04CapColor.png')

    # 05 does bruise
    createGraphBinaryValue(data, edibles, poisonous, 'does-bruise-or-bleed',
    ["Bruise or Bleed", "Doesn't"], 0.35, 
    'Does bruise or bleed by class', '05BruiseOrBleed.png')

    # 06 gill attachment
    createGraphEnumValue(data, edibles, poisonous,
    ['gill-attachment-empty', 'gill-attachment-a', 'gill-attachment-x', 'gill-attachment-d',
    'gill-attachment-e', 'gill-attachment-s', 'gill-attachment-p', 'gill-attachment-f'],
    ['Empty value', 'Adnate', 'Adnexed', 'Decurrent',
    'Free', 'Sinuate', 'Pores', 'None'], 0.35,
    'Gill attachment by class', '06GillAttachment.png')

    # 07 gill spacing
    createGraphEnumValue(data, edibles, poisonous,
    ['gill-spacing-empty', 'gill-spacing-c', 'gill-spacing-d', 'gill-spacing-f'],
    ['Empty value', 'Close', 'Distant', 'None'], 0.35,
    'Gill spacing by class', '07GillSpacing.png')

    # 08 gill color
    createGraphEnumValue(data, edibles, poisonous,
    ['gill-color-n', 'gill-color-b', 'gill-color-g', 'gill-color-r', 'gill-color-p', 'gill-color-u',
    'gill-color-e', 'gill-color-w', 'gill-color-y', 'gill-color-o', 'gill-color-k', 'gill-color-f'],
    ['Brown', 'Buff', 'Gray', 'Green', 'Pink', 'Purple',
    'Red', 'White', 'Yellow', 'Orange', 'Black', 'None'], 0.35,
    'Gill color by class', '08GillColor.png')
    
    # 09 stem height
    createGraphMetricalValue(data, edibles, poisonous, 'stem-height',
     'Stem height of poisonous and edible mushrooms', 'Mushroom', 'Height', '09StemHeight.png')

    # 10 stem width
    createGraphMetricalValue(data, edibles, poisonous, 'stem-width',
     'Stem width of poisonous and edible mushrooms', 'Mushroom', 'Width', '10StemWidth.png')

    # 11 stem root
    createGraphEnumValue(data, edibles, poisonous,
    ['stem-root-empty', 'stem-root-b', 'stem-root-s',
     'stem-root-c', 'stem-root-r', 'stem-root-f'],
    ['Empty value', 'Bulbous', 'Swollen',
     'Club', 'Rooted', 'None'], 0.35,
    'Stem root by class', '11StemRoot.png')

    # 12 stem surface
    createGraphEnumValue(data, edibles, poisonous,
    ['stem-surface-i', 'stem-surface-g', 'stem-surface-y', 'stem-surface-s', 
    'stem-surface-h', 'stem-surface-k', 'stem-surface-t', 'stem-surface-f'],
    ['Fibrous', 'Grooves', 'Scaly', 'Smooth',
    'Shiny', 'Silky', 'Sticky', 'None'], 0.35,
    'Stem surface by class', '12StemSurface.png')

    # 13 stem color
    createGraphEnumValue(data, edibles, poisonous,
    ['stem-color-n', 'stem-color-b', 'stem-color-g', 'stem-color-r', 'stem-color-p', 'stem-color-u',
    'stem-color-e', 'stem-color-w', 'stem-color-y', 'stem-color-l', 'stem-color-o', 'stem-color-k', 'stem-color-f'],
    ['Brown', 'Buff', 'Gray', 'Green', 'Pink', 'Purple',
    'Red', 'White', 'Yellow', 'Blue', 'Orange', 'Black', 'None'], 0.35,
    'Stem color by class', '13StemColor.png')

    # 14 veil type
    createGraphEnumValue(data, edibles, poisonous,
    ['veil-type-empty', 'veil-type-u'],
    ['Partial', 'Universal'], 0.35,
    'Veil type by class', '14VeilType.png')

    # 15 veil color
    createGraphEnumValue(data, edibles, poisonous,
    ['veil-color-n', 'veil-color-u', 'veil-color-e',
    'veil-color-w', 'veil-color-y', 'veil-color-k'],
    ['Brown', 'Purple','Red',
    'White', 'Yellow', 'Black'], 0.35,
    'Veil color by class', '15VeilColor.png')

    # 16 has ring
    createGraphBinaryValue(data, edibles, poisonous, 'has-ring',
    ["Has ring", "Doesn't"], 0.35, 
    'Does it has a ring by class', '16HasRing.png')

    # 17 ring type
    createGraphEnumValue(data, edibles, poisonous,
    ['ring-type-empty', 'ring-type-e', 'ring-type-r', 'ring-type-g', 'ring-type-l',
    'ring-type-p', 'ring-type-z', 'ring-type-m', 'ring-type-f'],
    ['Empty value', 'Evanescent', 'Flaring', 'Grooved', 'Large',
    'Pendant', 'Zone', 'Movable', 'None'], 0.35,
    'Ring type by class', '17RingType.png')

    # 18 spore print color
    createGraphEnumValue(data, edibles, poisonous,
    ['spore-print-color-n', 'spore-print-color-g', 'spore-print-color-r', 'spore-print-color-p',
     'spore-print-color-u', 'spore-print-color-w', 'spore-print-color-k'],
    ['Brown', 'Gray', 'Green', 'Pink',
    'Purple', 'White', 'Black'], 0.35,
    'Spore print color by class', '18SporePrintColor.png')

    # 19 habitat
    createGraphEnumValue(data, edibles, poisonous,
    ['habitat-g', 'habitat-l', 'habitat-m', 'habitat-p',
    'habitat-h', 'habitat-u', 'habitat-w', 'habitat-d'],
    ['Grasses', 'Leaves', 'Meadows', 'Paths',
    'Heaths', 'Urban', 'Waste', 'Woods'], 0.35,
    'Habitat by class', '19Habitat.png')

    # 19 season
    createGraphEnumValue(data, edibles, poisonous,
    ['season-s', 'season-u', 'season-a', 'season-w'],
    ['Spring', 'Summer', 'Autumn', 'Winter'], 0.35,
    'Season by class', '20Season.png')

    # heatmap best
    plt.figure(figsize=(6, 8))
    sns.heatmap(data.corr()[['class']].sort_values('class', ascending=False)[1:31], annot=True, vmin=-1, vmax=1)
    plt.title('Heat map of top 30 most correlated variables to being edible')
    plt.savefig('../Results/Analysis/heatMap30Best.png', bbox_inches='tight')
    plt.close()

    # heatmap worst
    plt.figure(figsize=(6, 8))
    sns.heatmap(data.corr()[['class']].sort_values('class', ascending=False)[-30:], annot=True, vmin=-1, vmax=1)
    plt.title('Heat map of top 30 least correlated variables to being edible')
    plt.savefig('../Results/Analysis/heatMap30Worst.png', bbox_inches='tight')
    plt.close()

    # collage
    # data.hist(figsize=(10, 5))
    # plt.tight_layout()
    # plt.savefig('../Results/Analysis/collageWithAllGraphs.png', bbox_inches='tight')
    # plt.close()