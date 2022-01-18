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
        d = 11 # TO DO: wat
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
    
    for i in Color._member_names_:
        data['veil-color-' + i] = np.where(data['veil-color'] == i, 1, 0)
    data.pop('veil-color')

    for i in VeilType._member_names_:
        data['veil-type-' + i] = np.where(data['veil-type'] == i, 1, 0)
    data.pop('veil-type')
    
    for i in Color._member_names_:
        data['spore-print-color-' + i] = np.where(data['spore-print-color'] == i, 1, 0)
    data.pop('spore-print-color')

    data['has-ring'] = [int(i == 't') for i  in data['has-ring']]

    for i in RingType._member_names_:
        data['ring-type-' + i] = np.where(data['ring-type'] == i, 1, 0)
    data.pop('ring-type')

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

def analyzeData(data):
    print(data.head())

    print(data.describe())

    # TO DO: wat hay setas con anillo pero sin anillo???
    # index = 0
    # for i in range(len(data['has-ring'])):
    #     if data['has-ring'][i] == True and RingType(data['ring-type'][i]) == RingType.f:
    #         index += 1
    # print(index)

    # data.hist(figsize=(10, 10))
    # plt.tight_layout()

    plt.figure(figsize=(6, 8))
    sns.heatmap(data.corr()[['class']].sort_values('class', ascending=False)[:30], annot=True, vmin=-1, vmax=1)
    plt.show()
