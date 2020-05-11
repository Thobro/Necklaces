from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from functools import reduce
import numpy as np
import math
import functions
import shape_read
from map_colors import map_colors
import random
from tqdm import tqdm
import configs

CONFIG = 'Europe'
THRESHOLD = 0
POINT_COUNT = 6
FILENAME_LOWRES = "Countries_110/ne_110m_admin_0_countries.shp"
FILENAME_HIGHRES = "Countries_110/ne_110m_admin_0_countries.shp"
#FILENAME_HIGHRES = "Countries_50/ne_50m_admin_0_countries.shp"
PLOT_POINTS = False
SHOW_TRIANGULATION = False
WATER = True
PLOT_WATER_POINTS = True

triangulation_cache = {}

fig = plt.figure(1, dpi=72)
fig.patch.set_visible(False)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_aspect('equal')

def trim(shape, trim_bounds):
    x_min, x_max, y_min, y_max = trim_bounds
    for i in range(len(shape)):
        shape[i] = [v for v in shape[i] if x_min <= v[0] <= x_max and y_min <= v[1] <= y_max]

    return [p for p in shape if len(p) != 0]

def prune(shape, record, threshold):
    '''Remove small islands etc.'''
    triangulation = triangulation_cache[record[configs.configs[CONFIG]['name_identifier']]]
    new_shape = []
    for i in range(len(shape)):
        area = functions.polygon_area(triangulation[i])
        if area > threshold:
            new_shape.append(shape[i])

    return new_shape

def prepare_shape_recs(shape_recs):
    shape_recs = [(trim(shape, configs.configs[CONFIG]['trim_bounds']), record) for shape, record in shape_recs if any([f(record) for f in configs.configs[CONFIG]['options']]) and all([f(record) for f in configs.configs[CONFIG]['requirements']])]
    shape_recs = [(shape, record) for shape, record in shape_recs if len(shape) != 0]
    shape_recs = [(shape, record) for shape, record in shape_recs if not any([f(record) for f in configs.configs[CONFIG]['exclude']])]
    return shape_recs

def get_colors_from_shape_recs(shape_recs):
    c = 0
    color_mapping = {}

    for shape1, rec1 in shape_recs:
        neighbors = [r[configs.configs[CONFIG]['name_identifier']] for (s, r) in shape_recs if r[configs.configs[CONFIG]['name_identifier']] in color_mapping and s != shape1 and functions.borders(s, shape1)]
        neighbor_colors = [color_mapping[n] for n in neighbors if n in color_mapping]
        while c in neighbor_colors:
            c += 1
        color_mapping[rec1[configs.configs[CONFIG]['name_identifier']]] = c % len(map_colors)

        c += 1
        c = c % len(map_colors)
    
    return color_mapping

def plot_shape_recs(shape_recs):
    color_mapping = get_colors_from_shape_recs(shape_recs)

    for shape, record in shape_recs:
        for polygon in shape:
            poly = Polygon(polygon)
            x,y = poly.exterior.xy
            ax.plot(x, y, color='000', alpha=1,
                linewidth=1, zorder=0)
            ax.fill(x, y, color=map_colors[record['MAPCOLOR13'] - 1], alpha=1,
                linewidth=0, zorder=0)

shape_recs = shape_read.shapefile_to_shape_recs(FILENAME_LOWRES)
shape_recs = prepare_shape_recs(shape_recs)


print("Computing triangulation...")
for shape, rec in tqdm(shape_recs):
    if not any([f(rec) for f in configs.configs[CONFIG]['exclude']]):
        triangulation = []
        for polygon in shape:
            tri = functions.triangulate_polygon(polygon)
            triangulation.append(tri)
        triangulation_cache[rec[configs.configs[CONFIG]['name_identifier']]] = triangulation

shape_recs = [(prune(shape, record, THRESHOLD), record) for shape, record in shape_recs]
split_dict = {'a': []}
for shape, rec in shape_recs:
    if not any([f(rec) for f in configs.configs[CONFIG]['show_but_exclude']]):
        if configs.configs[CONFIG].get('grouping'):
            if configs.configs[CONFIG]['grouping'](rec) not in split_dict:
                split_dict[configs.configs[CONFIG]['grouping'](rec)] = [(shape, rec)]
            else:
                split_dict[configs.configs[CONFIG]['grouping'](rec)].append((shape, rec))
        else:
            split_dict['a'].append((shape, rec))
    continue

plot_shape_recs(prepare_shape_recs(shape_read.shapefile_to_shape_recs(FILENAME_HIGHRES)))
all_triangulation = []
for region in triangulation_cache:
    for shape in triangulation_cache[region]:
        for part in shape:
            all_triangulation.append(part)

if WATER:
    water_sample = functions.sample_points_in_water(all_triangulation, 100, *configs.configs[CONFIG]['trim_bounds'])

point_sets = []
circles = []
for region in split_dict:
    point_sets_local = []
    for shape, rec in split_dict[region]:
        print(rec['NAME'])
        sample = functions.sample_shape(shape, rec, POINT_COUNT, triangulation_cache[rec[configs.configs[CONFIG]['name_identifier']]], THRESHOLD)
        point_sets_local.append(sample)
        point_sets.append(sample)
    
    if WATER:
        point_sets_local.append(water_sample)
    
    disc = functions.smallest_k_disc_facade(point_sets_local)
    circle = plt.Circle(disc[0], disc[1], fill=False, edgecolor="k", lw=3, clip_on=False)
    circles.append(circle)

if PLOT_WATER_POINTS:
    x, y = zip(*water_sample)
    ax.plot([x], [y], marker='o', markersize=3, c=(0, 0, 0), zorder=3)

if SHOW_TRIANGULATION:
    for polygon in all_triangulation:
        poly = Polygon(polygon)
        x,y = poly.exterior.xy
        ax.plot(x, y, color='000', alpha=1,
            linewidth=1, zorder=0)

if PLOT_POINTS:
    x, y = zip(*point_sets[0])
    ax.plot([x], [y], marker='o', markersize=3, c=(0, 0, 0), zorder=3)
    for i in range(1, len(point_sets)):
        x, y = zip(*point_sets[i])
        ax.plot([x], [y], marker='o', markersize=3, c=(0.3, 0.3, 0.3), zorder=3)

#p = [(2741942.593980968, 11254663.460523574), (2337158.5464477325, 11061415.644239938), (1368162.9922522276, 10058820.314567093), (1577845.8699682872, 10132199.41223928), (-495087.7700295333, 8007471.430739591), (-877472.2554764792, 7309193.80084587), (-1051482.4366690367, 7111058.242515617), (-913614.5090840559, 7223899.314557583), (1778148.009074803, 10137049.778159069), (-841696.1552608863, 7249913.532888901), (-550091.968056741, 7556087.425444108), (-608941.950185939, 7467266.424176965), (-378990.5652529128, 7710358.5454256125), (-1005865.1264422464, 6974171.00978223), (-1045306.2169000758, 6925749.749628562), (-864888.766385196, 7117117.13775868), (1319647.6375947641, 9542621.688592235), (-284449.451670331, 7682656.898076447), (-968297.9578384021, 6877412.666318495), (-917322.8335715282, 6919577.351739046), (-959792.0841048041, 6868664.3413127195), (-874196.7955282615, 6954673.547594839), (764642.0747289094, 8839869.947173905), (-717401.7045602828, 7099880.3742055185), (1603457.8910992567, 9831093.222730648), (-936961.6621835968, 6811814.844621719), (-362849.7252068528, 7469281.9005712), (-413060.4618443189, 7402824.547254713), (-775239.1841060659, 6977765.257042488), (-784005.2639235507, 6948323.877953197), (635416.0308774549, 8565607.559984297), (650900.0454548805, 8566662.162374424), (-384249.82086184376, 7218692.591996464), (2151521.504677547, 10445889.649174947), (-739839.9684621432, 6696141.626443358), (-326331.4959345665, 7135221.813742809), (2224353.2273025042, 10495058.908638228), (-166727.45888455177, 7071553.700153152), (-143599.22617537327, 6983490.771006832), (1239832.5826837616, 8967326.390942872), (1206340.8307768237, 8887676.011516415), (1868516.9770375597, 9884691.331654027), (-221697.71243494193, 6788031.806876771), (858274.1199483111, 8295488.662368771), (-1226048.0314873839, 5186503.366851057), (1590651.1246800323, 9347384.822375346), (-86271.16536862671, 6682575.996648833), (1119608.601287006, 8556708.085262459), (-1078377.750325565, 5165666.057136769), (5008.560039893317, 6657108.465910491), (-957208.4154521769, 5121628.319019839), (-937900.7501466727, 5142766.961131173), (1913280.1050471922, 9775866.48185627), (252824.0268008495, 6927778.834712029), (-965093.4053888388, 4938695.371612661), (-771375.9776599492, 5223543.072916439), (2391899.7801809805, 10608146.706939721), (-880211.5037823063, 5043694.556128193), (63107.481396923686, 6556998.227993867), (-845331.5488445258, 5020579.809716742), (-970139.5610816051, 4768526.7906963965), (-1006409.0112635563, 4703057.931135856), (-30657.544193100533, 6273905.552194595), (-753877.8672287803, 5070347.032215333), (-957599.295954493, 4735250.989001489), (-733760.5614927238, 5062889.097839664), (-753116.9335621586, 5008621.431361108), (-994731.1987058672, 4588905.525290194), (-923908.2164043119, 4669870.716576769), (108281.00387048081, 6368965.02535068), (-164961.7191848182, 5850542.077718057), (286633.9213280279, 6604356.338094146), (-952565.5201506946, 4417426.8257622905), (-163770.18544543663, 5763375.844217736), (-864336.9933229148, 4493050.307056444), (490123.02265889524, 6861493.264339544), (293528.96927803045, 6489654.969392121), (-391593.4797250945, 5240793.890202516), (-791600.8674922667, 4497666.917970528), (336987.9494228406, 6517462.553127845), (1418286.452652642, 8563348.616008582), (449546.18163189664, 6675852.465748898), (-294969.9216021467, 5275505.36770378), (452230.8193554883, 6655814.256395871), (450791.5165952431, 6649111.928972999), (109614.2714685566, 6007183.944921162), (390104.97272880067, 6522010.86593121), (579153.3106202674, 6877424.573715952), (1111932.3698805969, 7900387.6699126875), (398375.9549589547, 6481464.937766836), (439195.0542620624, 6536245.760402299), (1703102.817125901, 9043750.58861775), (533152.6857218584, 6667701.552386519), (259773.36974241785, 6131798.909002125), (1124443.9620645153, 7826444.652903766), (456349.2275429534, 6497308.62847699), (1104178.987099977, 7774444.6755573), (636474.9097084056, 6830230.424364971), (456399.6761083839, 6474715.449022391), (557777.3671971653, 6652408.678406868), (604975.0935229053, 6740996.266035279), (770445.0871936004, 7034088.624941737), (634420.4360927683, 6756294.408921541), (355367.7085164467, 6195148.873186365), (603824.9206682584, 6671161.308395453), (1008901.692310006, 7486879.783175116), (-91131.51920073187, 5286641.680584852), (-302472.3523531723, 4869669.482573842), (866595.246194295, 7168426.854538272), (603649.4329519481, 6620279.185394994), (775964.7982542696, 6956844.760716207), (630772.9641074669, 6657328.138824262), (996038.6269733576, 7409702.89701897), (957562.119466732, 7309955.917906049), (523782.5981553961, 6411352.675759334), (571276.7854003005, 6504966.738852593), (695456.5499866308, 6759801.487606137), (-278236.6068990747, 4795601.974902555), (1120282.593278424, 7635524.422336369), (1686310.3577513723, 8831656.283119727), (-201777.31557088014, 4877730.520987694), (586348.90689987, 6452135.097527379), (832410.5114144054, 6923053.798002011), (1063518.5428656123, 7402309.004415989), (595035.5135907625, 6393235.396677941), (654768.9799382539, 6504696.443330798), (641958.8157631254, 6468295.206684789), (667306.1037034036, 6506714.546627045), (1073876.3777007493, 7385888.013633624), (-264820.0007776428, 4555501.172569323), (746901.1155776257, 6640980.411005576), (628043.3235972716, 6382007.155641485), (954397.3235216053, 7079954.595255754), (660640.4985509939, 6439696.902634379), (615317.5368478983, 6337374.808588325), (650896.61125735, 6401254.585803018), (625650.6653049903, 6335363.114895821), (645326.3906509054, 6372550.058351816), (964068.1672701673, 7070020.496948044), (638920.567142581, 6357399.016077487), (661954.1058489529, 6396358.078717952), (634416.806199614, 6333414.008228668), (636394.23707888, 6331792.983769712), (657359.7242849595, 6361795.797375972), (654118.3390701518, 6324046.670516483), (655605.9413055294, 6325938.538496385), (-20445.135643809488, 4857701.6738982815), (963659.4325259939, 6995728.561969899), (549900.5032615439, 6067128.056955898), (690188.8658075877, 6371238.836683396), (667605.7684731934, 6321246.167273546), (1889460.6330030863, 9133712.95422114), (672956.9700352551, 6323250.684493426), (680113.0286898206, 6321275.444412311), (695726.2340008413, 6353806.905685606), (467981.7486375185, 5846351.92311253), (683711.5314507438, 6322060.773483592), (-127611.70023410814, 4557742.1726388), (-185596.9072942519, 4420265.957182614), (1667378.020175038, 8578665.074444454), (665100.6225868837, 6249134.434455894), (-84791.9209160393, 4550874.793370711), (1227552.2437422043, 7480997.909109111), (688911.4152939675, 6196455.642184697), (429854.34278334695, 5603990.466409161), (703821.5643640303, 6191908.642939316), (1271136.9067479759, 7471022.102574794), (1394989.065181761, 7757725.809437273), (330117.561329668, 5238016.059033191), (1276072.7601008601, 7432417.299009787), (1471188.8073209978, 7899355.907125163), (869390.5941340025, 6415394.054721881), (1079203.8647588294, 6898245.16177658), (976646.5682379883, 6643562.344447112), (406906.46612422715, 5218417.91859706), (441200.5302727193, 5286420.417529013), (1945129.2819236796, 9028909.776175048), (704988.4498964852, 5838151.630104586), (725407.3518212948, 5851401.177919354), (1324807.27957938, 7343186.318312397), (1208721.6164639727, 6991513.573173044), (786454.6690776475, 5870978.971578681), (600618.5566421488, 5360071.673661437), (828299.7999758457, 5897776.877364651), (1182133.2381111253, 6771710.690372882), (2237369.0648362814, 9721238.860449735), (1102561.0517967222, 6481245.917121277), (1865653.8226530543, 8604827.794347495), (838025.9583564112, 5738084.465217616), (849749.3987047778, 5762363.986495117), (911484.9721003593, 5869842.763765127), (937082.0595647233, 5903215.345176055), (889655.7651455582, 5771072.413230502), (918041.9853912278, 5844887.887066313), (995018.6433087512, 5948439.204961728), (969833.0438619597, 5853796.850912946), (1114471.045897638, 6237203.643260935), (1053952.7315544286, 5948093.516806426), (1023193.519034173, 5849766.956805853), (1067981.1333293521, 5947295.439805544), (922789.3850847752, 5509122.779809284), (898489.7635818498, 5322215.872349562), (1089851.4142773205, 5839234.81623809), (1302289.6031086487, 6429090.017014369), (1254608.52001729, 6231581.692527037), (1176273.0116505516, 5972239.089599075), (1151058.6571069486, 5887755.722904313), (1239290.107600485, 6154611.026743396), (1391105.3151643216, 6453283.302035543), (1155204.811022897, 5679630.165646816), (1339604.8027806086, 6213429.999382175), (1114532.7350403708, 5474723.87722738), (963242.5483540887, 4872818.501586297), (1564693.0195143041, 6836156.049201054), (1145975.4129790834, 5416052.334836483), (1605665.574135765, 6914983.968526889), (1333393.6416488935, 5977939.174760594), (1327763.6176100352, 5956932.262271205), (960167.5801069215, 4725713.384668344), (1322585.2637659442, 5916048.504605416), (1192357.9618975334, 5408441.943801253), (1634232.283581743, 6930096.483335278), (1489004.702484617, 6394869.333393503), (998199.8502157597, 4692419.6672892915), (1303696.5428960756, 5624186.0981822545), (1657031.899419261, 6887762.663086828), (1530798.9878887092, 6372540.71600174), (1526503.6637101646, 6239234.078325101), (1441637.105910078, 5922044.171080197), (2092579.2589609206, 8388659.29907632), (1260658.5308204796, 5178197.921116891), (2204849.612989922, 8835846.734640162), (1644599.3029786975, 6567858.162972523), (1595166.4337751437, 6200302.887476551), (1653534.1340278923, 6360900.695190635), (1607820.0450408733, 6080312.925010522), (1548319.3810431205, 5773626.197370031), (1428530.529892489, 5289966.5305154165), (1570888.2483904, 5818385.939703822), (1539633.8515066975, 5661773.350135538), (1714298.5602149128, 6383835.123550503), (1428816.478898624, 5194167.313741445), (1551831.8871123355, 5656318.409543735), (1580215.293494136, 5766813.723737775), (1613192.0921612938, 5896517.43451668), (1699732.2713456678, 6245065.218925889), (1760276.661511973, 6446124.961911818), (1670284.7975500529, 6057505.6107282825), (1722978.9575892151, 6271693.18358114), (2272515.7936066696, 8794818.15973127), (1619181.0301069538, 5819048.272931334), (1462021.7170189614, 5153647.7235131925), (1616303.4950337643, 5789919.2070217915), (1601485.534796379, 5712755.090990019), (1653305.7996550892, 5918532.579247563), (1637687.7302840564, 5826348.256464955), (1759848.6467976642, 6345122.065599995), (1628765.3332351006, 5764665.313108691), (1706166.0328778906, 6086844.6278268555), (1635927.969418677, 5779488.680322406), (1610482.8906075524, 5653272.672106177), (1667613.2182721351, 5841139.73265499), (1867811.979910479, 6702901.750890856), (1647439.039079913, 5697544.879142918), (1700952.6753021872, 5889982.372447697), (1805307.92015333, 6334453.366813049), (1824900.7778778537, 6407553.441801798), (1638505.9426504232, 5529785.684038634), (1691897.3747076893, 5763001.255302994), (1816304.1018425757, 6319547.803958064), (1674991.3190289966, 5662821.686765875), (1615937.8157783896, 5332150.481827301), (1792472.0888942424, 5984806.083799345), (1725891.9494452782, 5659602.373333108), (1748469.2434994383, 5747119.359489085), (1896331.5851741433, 6464152.61081865), (1616078.9395885447, 5113079.108134875), (1804749.6491353167, 6003484.5768960025), (2635288.872407803, 10561400.733565073), (1800831.6097757535, 5907122.874053654), (1805742.7042837401, 5866859.58666461), (1799348.7679624485, 5814535.600727517), (1808957.9849936033, 5842466.412280679), (1734741.4957318967, 5469467.67616864), (1762220.7290245083, 5559226.656430666), (2053852.636953172, 7035693.73423645), (1774207.4322731066, 5544397.104935949), (2077057.5023158062, 7043471.582473059), (1828575.5750958622, 5671015.111958158), (1812006.0438851486, 5486738.8494302295), (1893387.0575244685, 5910994.798779403), (1950388.5322194204, 6166695.435773068), (2141922.3764846423, 7213435.196574606), (1969601.07841725, 6178450.607325959), (1898357.192892858, 5783229.956014229), (1976405.0109258976, 6133044.006233491), (1998517.7765908942, 6249028.616503822), (1977054.1406624191, 6059285.716801461), (1917568.7025326672, 5650074.922793701), (1993895.5562703519, 5998563.044499435), (2637687.9005466783, 10382628.410038842), (1933730.9161515757, 5622228.964030551), (1979405.5190930057, 5864715.833686104), (1985687.4934003053, 5870952.734141953), (1956775.3470993699, 5694487.590393632), (2253475.052255559, 7540229.607691086), (1924384.587539249, 5448613.134177639), (1983460.6826434946, 5728414.009644386), (1931244.1905742746, 5397846.864635847), (1935495.1341085497, 5393135.954367914), (2038690.322371243, 6016844.035554282), (2046112.0302407865, 6061416.356229175), (2245895.9367247806, 7349686.867027859), (2100256.5426300913, 6321308.181832652), (1996304.4720843642, 5606787.900866177), (2021073.9107373452, 5745699.901918434), (2048939.3855625377, 5921361.0094066905), (2102070.0919290376, 6070944.583457005), (1921113.8828301365, 4892864.038575062), (1985471.1722366854, 5271627.7641704865), (2039145.7147320083, 5568293.90134385), (1992756.8772986073, 5245930.053732042), (2237729.000396358, 6906758.090448843), (2257563.808101844, 7040941.316467973), (2027889.5593130002, 5368712.858745309), (2092309.0065139583, 5786444.864742511), (2170080.5718005374, 6308585.490671798), (2009293.858126528, 5210769.749041263), (2035334.4308056915, 5312669.943002987), (2368256.496136985, 7653352.793198582), (2071407.5149955014, 5361861.702712881), (2064975.2667382162, 5232028.1571916975), (2078014.5377523126, 5322305.921904012), (2113041.3218988506, 5539137.5729637), (2078057.7454986516, 5286671.404247707), (2066343.2177282395, 5194402.31616851), (2205431.997790592, 6206335.593499106), (2132240.331215689, 5635881.425034129), (2113038.895890302, 5379667.622042523), (2103617.3715494373, 5292885.388476375), (2108749.7123847515, 5310483.947798522), (2138730.235406171, 5470969.525450375), (2118172.770992467, 5257573.008543325), (2135087.311949153, 5303194.791090795), (2210307.976768894, 5862903.357923448), (2398766.207551568, 7448908.631950866), (2142733.525658204, 5291551.840085923), (2231739.640619844, 5989580.991240981), (2175351.7928684796, 5541507.76467433), (2203160.1981981113, 5722943.963880335), (2156505.855068287, 5304697.29516434), (2155201.1446678573, 5291111.997045708), (2249351.9740613615, 6020304.902843958), (2187674.7153770244, 5512594.356751252), (2167582.0470405407, 5323608.381149139), (2458446.398699837, 7866664.008914589), (2214267.551334439, 5655509.91144478), (2173964.141618206, 5315351.744051489), (2213306.531631971, 5614468.357509821), (2125323.7417659364, 4856473.82186568), (2186828.1797338314, 5248721.093260403), (2304426.7192449104, 6240802.470748928), (2188989.2155903177, 5246234.491624904), (2272142.0272498336, 5915018.348609825), (2182978.5135793965, 5106516.880538756), (2175039.4774819533, 5038967.618873875), (2366957.224208056, 6689982.132747044), (2201218.194737543, 5167768.204202847), (2214290.940680502, 5211914.551907862), (2462369.244249275, 7522836.1395116495), (2199551.599727944, 5003946.572293593), (2492348.9805537257, 7766843.379490377), (2379880.683867405, 6571553.607334871), (2241160.721263406, 5213363.2431019265), (2225466.367207362, 5053838.7524042195), (2243402.111412565, 5208848.890213172), (2288206.7533520185, 5611766.5168135455), (2615440.7951868647, 9215823.198647765), (2362245.6879894887, 6301492.914904657), (2234735.174179203, 5100182.0872982275), (2230710.097913831, 5041808.13799149), (2262530.8069150923, 5279275.059301727), (2316732.495969791, 5767934.380349097), (2249556.519370183, 5123180.595535952), (2226522.1180351824, 4890074.271741276), (2301082.031911928, 5565746.899345462), (2274614.785794195, 5254998.760872302), (2278076.3540399754, 5240616.555611236), (2270409.6682267026, 5168871.251091918), (2370729.6871432276, 6112706.967503546), (2230822.589009527, 4773512.383951781), (2364897.80995176, 6018039.968583044), (2309742.460603434, 5471765.938565773), (2245175.2156062415, 4848826.667113956), (2299873.4133823775, 5357430.938581377), (2294031.423355137, 5215446.1617189655), (2283696.16854168, 5103567.275875895), (2414352.3162517073, 6395130.890735821), (2343805.624358177, 5650936.448227042), (2296615.9390106117, 5175505.493400286), (2328232.2358833067, 5470237.747211359), (2347692.565202111, 5633715.754005656), (2317170.15547844, 5314883.583302837), (2289000.9754718873, 5040532.085984945), (2315465.468125412, 5260880.814096748), (2339192.511277313, 5491646.510030384), (2320732.636658011, 5305236.414959719), (2456927.566914073, 6757057.821887728), (2324258.035541703, 5312768.058467108), (2281275.1225748616, 4869457.4658176275), (2317073.701903819, 5213752.10905511), (2316384.1389558385, 5175171.145651147), (2419641.5217559757, 6229737.2000357), (2298612.1689592698, 4921152.643196896), (2349605.821238564, 5428365.114802684), (2319853.4871161445, 5103239.270468628), (2333514.9751490853, 5228354.465355036), (2337827.513657515, 5216105.384679473), (2318927.76764313, 4971685.036313252), (2430864.6298313895, 6165926.178974919), (2347217.723992257, 5211599.704987295), (2443151.7435995224, 6216693.0423230035), (2339616.879413027, 5052097.296553847), (2320877.3577231513, 4815922.614297123), (2451811.512022148, 6237626.106337667), (2532386.7428999883, 7235446.188943237), (2400650.2967994083, 5573096.04198878), (2594802.395232799, 8131863.374053166), (2378944.025299823, 5310289.904355914), (2317524.547446486, 4636820.771583892), (2487353.236573209, 6539189.301053947), (2355755.7771451646, 4994991.146655974), (2554393.027923521, 7449087.109720268), (2373842.789755696, 5157360.210626021), (2370381.960625722, 5067879.271635927), (2347174.3966955096, 4798698.245150804), (2410376.918595424, 5489060.515182808), (2365281.411232522, 4951352.007978329), (2406790.4284563866, 5360244.415814014), (2376360.299601081, 4973584.896219281), (2576093.4994786154, 7561489.645261109), (2460774.2755547604, 5900056.165447957), (2376306.3354107286, 4592019.698563353), (2421676.0594215775, 5027535.662429644), (2585730.5011077817, 7293988.291568965), (2440929.027786061, 5084490.356157702), (2427954.6255494813, 4898258.889974256), (2526738.392430963, 6172659.269218524), (2431618.3166487655, 4883369.894516048), (2458862.684376281, 5084337.351580577), (2490083.751351475, 5498953.563509436), (2649775.374114359, 8164321.83678112), (2487900.7204454527, 5364730.145957666), (2474358.0186358453, 5113600.624435334), (2463474.9565205546, 4938277.970813988), (2538828.894670764, 5854890.466874605), (2653803.0253103254, 7903619.851300526), (2492999.4247447955, 5059069.215125592), (2654667.266535175, 7844415.857285226), (2655472.5009957273, 7811745.315366784), (2466932.40804255, 4554270.311932311), (2530868.4313172717, 5418724.094800439), (2498100.6057293448, 4927393.295271111), (2608743.113317827, 6653024.536134394), (2520809.667261199, 5133726.000550407), (2649419.0054168627, 7380741.353048388), (2525124.7197446856, 5075633.5436207745), (2472590.7307540327, 4310513.872246978), (2481463.713345647, 4386455.299270184), (2511616.1482076496, 4784419.332101931), (2566937.2783046435, 5611362.611520422), (2561975.6547061754, 5110838.725485953), (2599369.6065942533, 5747895.940250502), (2677661.5486251106, 7354840.342943442), (2572552.586347329, 5043291.007789707), (2681214.0564024397, 7280487.995709491), (2678587.3473035335, 7184022.930805616), (2605958.741102055, 5048968.706780822), (2688625.506045641, 6785555.328547204), (2748229.308786577, 10237799.696120353), (2727089.977643959, 7754614.455646712), (2737823.1829860057, 8244445.068530558), (2726148.2187361685, 7505706.639169414), (2737973.5742542204, 7932776.156022107), (2691956.1926876456, 6098038.310150257), (2747785.6166999396, 8245235.838556707), (2750368.557542795, 8233164.771373889), (2694188.8145989315, 5782732.102605081), (2680312.5937251933, 5200861.512331964), (2754301.671875053, 7961283.109965129), (2690119.0223991717, 5253958.659893174), (2709391.728584823, 5625970.198127069), (2730613.082371866, 6155667.043217218), (2770041.773302168, 8145220.925339317), (2778354.55014786, 8174016.466323497), (2722005.0263587767, 5058830.096752122), (2766510.2886272273, 6766930.733016496), (2749479.9082905767, 5055535.8932557255), (2779280.5287532196, 6131952.320234678), (2800690.9016175764, 6477456.444448643), (2808834.492303114, 6893402.062224633), (2764015.5782794068, 4158394.63464271), (2818344.638034465, 7252326.169507461), (2817987.3489376414, 8215265.751482063), (2822406.109441853, 7702143.764130693), (2825857.3700568955, 7445053.319328821), (2830983.9162321244, 7582145.071050336), (2825888.09874223, 5182767.90424862), (2842545.693744456, 7495519.965336255), (2819839.8052765527, 4176986.7905622805), (2856209.853303527, 5677739.48773594), (2853846.678810321, 7439200.582109412), (2859129.9762703557, 7932514.554240411), (2880273.7137692226, 5451649.044349638), (2843612.212291408, 8899388.190323245), (2892547.293731344, 5365339.0468299), (2894264.7570989905, 5073008.1349310335), (2890091.372764781, 6981926.718553927), (2879077.205694656, 7801422.770344055), (2916731.6085836436, 5396384.574964032), (2866402.0116105354, 8545067.989086231), (2921774.231466318, 5190174.119239845), (2923765.5704429615, 5250577.404064826), (2933343.1361042773, 5448251.586460448), (2938868.2920576413, 6233313.94910691), (2910977.880107957, 7675214.256040422), (2978654.7720610998, 5048493.904281003), (2983678.2322022766, 5434214.405261065), (2965100.2193669425, 6745855.043747237), (2998713.8220358347, 5414239.037570819), (2962783.1284938008, 7517950.621292933), (2993011.4943456557, 6902588.4762651045), (2968545.5854928135, 7673446.500833564), (2983101.3849211787, 7491815.004512552), (3030670.0414466457, 6505367.1522814045), (2965981.7060798006, 8049173.855979939), (3095779.8220024314, 5315235.080839537), (3075069.5637500165, 6009150.556608042), (2994726.0266472325, 7847846.2409008695), (3090397.898924726, 5835119.382152713), (3097604.7302174494, 5963792.335349562), (3011384.5831532422, 7775126.132422018), (3056913.187728533, 7106854.693755054), (3119731.740444001, 6055814.96095202), (3042487.039392128, 7545789.814629638), (3127947.4686058336, 6108358.200071892), (3136914.3033298487, 6070877.631833641), (3148942.039359569, 5829878.608242301), (3145743.0489783743, 5900629.428808207), (3174598.6047077538, 5782158.325089816), (3187446.3188567553, 5803977.921247962), (3197057.2540162704, 5765564.135091431), (3191307.138295199, 5921273.491089903), (3076045.7610545154, 7694795.739321363), (3230788.687175433, 5409148.612277572), (3220547.912600642, 5826506.18107951), (3221194.0226585288, 5856470.352041555), (3223998.3270229273, 5919538.673297988), (3235702.8363698185, 5973914.68474433), (3096527.4572342904, 7822005.138625017), (3121636.831956857, 7525738.323203139), (3296921.683544004, 5247122.839886261), (3273217.037686361, 5963134.103806212), (3225160.25856267, 6941518.705469985), (3335568.3268988314, 5858945.046344896), (3395518.0640362827, 6071374.396403882), (3293549.1744664013, 7080899.452763161), (3028790.443465945, 9299316.533281812), (3279219.3097923873, 7425241.125065638), (3485967.5751343546, 5879528.208468937), (3388584.273630022, 6830734.036768633), (3340981.3114188793, 7259642.18522345), (3182668.4048797917, 8619013.439006258), (3395790.301847434, 7184449.260140639), (3578903.88464884, 5845363.164172397)]
#x, y = zip(*p)
#ax.plot([x], [y], marker='o', markersize=3, c=(1, 1, 1), zorder=3)


for c in circles:
    ax.add_artist(c)

plt.savefig("filename.pdf", dpi=72, bbox_inches = 'tight')
plt.show()
