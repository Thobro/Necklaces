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
        'trim_bounds': (9.5*10**6, 1.6*10**7, -1.7*10**6, 3.5*10**6),
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
        'trim_bounds': (-1*10**7, -3.2*10**6, -8*10**6, 2*10**6),
        'exclude': [
        ],
        'show_but_exclude': [
        ],
        'name_identifier': 'NAME',
    },
    'America': {
        'options': [
            lambda r: r['SUBREGION'] == "Northern America",
            lambda r: r['SUBREGION'] == "Central America",
        ],
        'requirements': [
            lambda r: r['POP_EST'] >= 10000
        ],
        'trim_bounds': (-2*10**7, -1*10**6, 0, 1.9*10**7),
        'exclude': [
            lambda r: r['NAME'] == 'Greenland',
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
    'AfricaPhysical': {
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
    'NL': {
        'options': [
            lambda r: r['admin'] == "Netherlands",
        ],
        'requirements': [
        ],
        'exclude': [
        ],
        'show_but_exclude': [
        ],
        'trim_bounds': (0*10**8, 1*10**6, -1*10**8, 1*10**8),
        'name_identifier': 'name',
    },
}
