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
        'trim_bounds': (-4*10**6, 6.3*10**6, 3.4*10**6, 12*10**6),
        'exclude': [
            lambda r: r['NAME'] == 'Svalbard Is.',
            
        ],
        'show_but_exclude': [
            lambda r: r['NAME'] == 'Iceland',
        ],
        'name_identifier': 'NAME',
    },
    'SEAsia': {
        'options': [
            lambda r: r['SUBREGION'] == "South-Eastern Asia",
        ],
        'requirements': [
            lambda r: r['POP_EST'] >= 1000
        ],
        'trim_bounds': (-1*10**8, 1*10**8, -1*10**8, 1*10**8),
        'exclude': [
        ],
        'show_but_exclude': [
        ],
        'name_identifier': 'NAME',
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
        'exclude': [
        ],
        'show_but_exclude': [
        ],
        'name_identifier': 'NAME',
    },
    'SAmerica': {
        'options': [
            lambda r: r['SUBREGION'] == "South America",
        ],
        'requirements': [
            lambda r: r['POP_EST'] >= 10000
        ],
        'trim_bounds': (-1*10**8, 1*10**8, -1*10**8, 1*10**8),
        'exclude': [
        ],
        'show_but_exclude': [
        ],
        'name_identifier': 'NAME',
    },
    'USA': {
        'options': [
            lambda r: r['region'] == "South",

        ],
        'requirements': [
        ],
        'trim_bounds': (-1*10**8, 1*10**8, -1*10**8, 1*10**8),
        'exclude': [
            lambda r: r['name'] == 'Alaska',
            lambda r: r['name'] == 'Hawaii'
        ],
        'show_but_exclude': [
        ],
        'name_identifier': 'name',
        'grouping': lambda r: {'South': 'a', 'Midwest': 'b', 'Northeast': 'c', 'West': 'd',}[r['region']],
    },
}
