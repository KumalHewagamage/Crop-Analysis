# Thresholds and weights
THRESHOLDS = {
    'black_spots': [(0,0), (1,1), (2,3), (4, 999)],   # (min,max) for levels 0..3
    'holes'      : [(0,0), (1,2), (3,4), (5, 999)],
    'white_fungus':[(0,0), (1,1), (2,3), (4, 999)],
}
WEIGHTS = {'black_spots': 0.3, 'holes': 0.4, 'white_fungus': 0.3}

def count_to_level(count, thresholds):
    """Map count (int) to level 0..3 using thresholds list of (min,max)."""
    for level, (mn, mx) in enumerate(thresholds):
        if mn <= count <= mx:
            return level
    return len(thresholds)-1

def grade_pineapple(bs_count, holes_count, wf_count):
    lb = count_to_level(bs_count, THRESHOLDS['black_spots'])
    lh = count_to_level(holes_count, THRESHOLDS['holes'])
    lw = count_to_level(wf_count, THRESHOLDS['white_fungus'])
    sraw = (WEIGHTS['black_spots'] * lb +
            WEIGHTS['holes'] * lh +
            WEIGHTS['white_fungus'] * lw)    # in 0..3
    s = sraw / 3.0
    if s < 0.25:
        grade = 'A'   # Excellent
    elif s < 0.5:
        grade = 'B'   # Good
    elif s < 0.75:
        grade = 'C'   # Fair
    else:
        grade = 'D'   # Poor
    return {'grade': grade, 'score': s, 'levels': (lb, lh, lw), 'sraw': sraw}

# # Example usage:
# print(grade_pineapple(0,0,0))      # expected A
# print(grade_pineapple(2,3,1))      # moderate -> likely C
# print(grade_pineapple(1,6,0))      # many holes -> C or D