from lunaproof.dem_io import boxes_overlap

TILES = {   # name: (lat, lon_east, role)
 'serenitatis': (28, 18, 'train'), 'imbrium': (33, 344.4, 'train'), 'fecunditatis': (-8, 51, 'train'),
 'nubium': (-21, 328, 'train'), 'procellarum': (0, 300, 'train'), 'descartes_hl': (-12, 16, 'train'),
 'farside_a': (0, 200, 'train'), 'farside_b': (-30, 140, 'train'), 'hadley': (26, 3, 'train'),
 'tranquillitatis': (8.5, 31.4, 'val'), 'south_highland': (-35, 25, 'val'),
 'tycho': (-43.3, 348.9, 'test'), 'copernicus': (9.6, 339.9, 'test'),
 'aristarchus': (23.7, 312.5, 'test'), 'orientale': (-19, 265, 'test'),
}

SIZE = 640
PPD = 64
deg = SIZE / PPD

def test_no_leakage_between_roles():
    names = list(TILES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = TILES[names[i]], TILES[names[j]]
            if a[2] != b[2]:
                assert not boxes_overlap(a, b, deg + 1), f"Role overlap detected between {names[i]} ({a[2]}) and {names[j]} ({b[2]})"
