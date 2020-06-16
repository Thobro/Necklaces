import configs
CONFIG = 'NL'

def trim(shape, trim_bounds):
    x_min, x_max, y_min, y_max = trim_bounds
    for i in range(len(shape)):
        shape[i] = [v for v in shape[i] if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max]

    return [p for p in shape if len(p) != 0]

def prepare_shape_recs(shape_recs):
    shape_recs = [(trim(shape, configs.configs[CONFIG]['trim_bounds']), record) for shape, record in shape_recs if any([f(record) for f in configs.configs[CONFIG]['options']]) and all([f(record) for f in configs.configs[CONFIG]['requirements']])]
    shape_recs = [(shape, record) for shape, record in shape_recs if len(shape) != 0]
    shape_recs = [(shape, record) for shape, record in shape_recs if not any([f(record) for f in configs.configs[CONFIG]['exclude']])]
    return shape_recs