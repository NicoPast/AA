from enum import Enum

import matplotlib.pyplot as plt

import seaborn as sns

#region Enums
class CapShape(Enum):
        empty = 0
        b = 1 
        c = 2
        x = 3
        f = 4
        s = 5
        p = 6
        o = 7

class Surface(Enum):
        empty = 0
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
        empty = 0
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
        empty = 0
        g = 1
        l = 2
        m = 3
        p = 4
        h = 5
        u = 6
        w = 7
        d = 8

class Season(Enum):
        empty = 0
        s = 1
        u = 2
        a = 3
        w = 4
#endregion

def prepareData(data):
    data['class'] = [int(i == 'e') for i  in data['class']]

    data['cap-shape'] = [CapShape[i].value for i in data['cap-shape']]
    data['cap-surface'] = [Surface[i].value for i in data['cap-surface']]
    data['cap-color'] = [Color[i].value for i in data['cap-color']]

    data['does-bruise-or-bleed'] = [int(i == 't') for i  in data['does-bruise-or-bleed']]

    data['gill-attachment'] = [GillAttach[i].value for i in data['gill-attachment']]
    data['gill-spacing'] = [GillSpacing[i].value for i in data['gill-spacing']]
    data['gill-color'] = [Color[i].value for i in data['gill-color']]

    data['stem-root'] = [StemRoot[i].value for i in data['stem-root']]
    data['stem-surface'] = [Surface[i].value for i in data['stem-surface']]
    data['stem-color'] = [Color[i].value for i in data['stem-color']]

    data['veil-type'] = [VeilType[i].value for i in data['veil-type']]
    data['veil-color'] = [Color[i].value for i in data['veil-color']]

    data['has-ring'] = [int(i == 't') for i  in data['has-ring']]
    data['ring-type'] = [RingType[i].value for i in data['ring-type']]
    
    data['spore-print-color'] = [Color[i].value for i in data['spore-print-color']]

    data['habitat'] = [Habitat[i].value for i in data['habitat']]
    
    data['season'] = [Season[i].value for i in data['season']]
    
    # unique = list(dict.fromkeys(data['season']))
    # print(unique)

def analyzeData(data):
    print(data.head())

    print(data.describe())

    # TO DO: wat hay setas con anillo pero sin anillo???
    index = 0
    for i in range(len(data['has-ring'])):
        if data['has-ring'][i] == True and RingType(data['ring-type'][i]) == RingType.f:
            index += 1
    print(index)

    data.hist(figsize=(15, 10))
    plt.tight_layout()

    plt.figure(figsize=(6, 8))
    sns.heatmap(data.corr()[['class']].sort_values('class', ascending=False), annot=True, vmin=-1, vmax=1)
    plt.show()
