configs = {
    'Europe': {
        'options': [
            lambda r: r['SUBREGION'] == "Northern Europe",
            lambda r: r['SUBREGION'] == "Western Europe",
            lambda r: r['SUBREGION'] == "Southern Europe",
            lambda r: r['SUBREGION'] == "Eastern Europe"
        ],
        'requirements': [
            lambda r: r['POP_EST'] >= 1000
        ],
        'trim_bounds': (-1*10**7, 6.3*10**6, -1*10**8, 12*10**6),
        'exclude': [
            lambda r: r['NAME'] == 'Svalbard Is.'
        ],
        'show_but_exclude': [
            lambda r: r['NAME'] == 'Iceland',
        ],
    },
    'SEAsia': {
        'options': [
            lambda r: r['SUBREGION'] == "South-Eastern Asia",
        ],
        'requirements': [
            lambda r: r['POP_EST'] >= 1000
        ],
        'trim_bounds': (-1*10**8, 1*10**8, -1*10**8, 1*10**8),
    },
    'Africa': {
        'options': [
            lambda r: r['SUBREGION'] == "Northern Africa",
            lambda r: r['SUBREGION'] == "Western Africa",
            lambda r: r['SUBREGION'] == "Southern Africa",
            lambda r: r['SUBREGION'] == "Eastern Africa",
            lambda r: r['SUBREGION'] == "Middle Africa",
        ],
        'requirements': [
            lambda r: r['POP_EST'] >= 0
        ],
        'trim_bounds': (-1*10**8, 1*10**8, -1*10**8, 1*10**8),
    },
    'SAmerica': {
        'options': [
            lambda r: r['SUBREGION'] == "South America",
        ],
        'requirements': [
            lambda r: r['POP_EST'] >= 10000
        ],
        'trim_bounds': (-1*10**8, 1*10**8, -1*10**8, 1*10**8),
    },
}
