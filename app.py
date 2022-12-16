import streamlit as st
import pathlib
import random
from functools import reduce
from collections import defaultdict
import pandas as pd
import folium
import shapely
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import sklearn.cluster
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests
import pandas as pd
import xmltodict
import json
import pydeck as pdk
import geopandas as gpd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Polygon, Point

#Pydeck 사용을 위한 함수 정의
import geopandas as gpd 
import shapely # Shapely 형태의 데이터를 받아 내부 좌표들을 List안에 반환합니다. 
def line_string_to_coordinates(line_string): 
    if isinstance(line_string, shapely.geometry.linestring.LineString): 
        lon, lat = line_string.xy 
        return [[x, y] for x, y in zip(lon, lat)] 
    elif isinstance(line_string, shapely.geometry.multilinestring.MultiLineString): 
        ret = [] 
        for i in range(len(line_string)): 
            lon, lat = line_string[i].xy 
            for x, y in zip(lon, lat): 
                ret.append([x, y])
        return ret 
def multipolygon_to_coordinates(x): 
    lon, lat = x[0].exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)] 
def polygon_to_coordinates(x): 
    lon, lat = x.exterior.xy 
    return [[x, y] for x, y in zip(lon, lat)]

st.title('제주도 이동형 전기차 충전 서비스')
df_result = pd.read_csv('데이터', index_col = 0)

def generate_candidate_sites(points,M=100):
    '''
    Generate M candidate sites with the convex hull of a point set
    Input:
        points: a Numpy array with shape of (N,2)
        M: the number of candidate sites to generate
    Return:
        sites: a Numpy array with shape of (M,2)
    '''
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([random.uniform(min_x, max_x),
                             random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

def generate_candidate_sites(df_result_fin,M=100):
    from shapely.geometry import Polygon, Point
    sites = []
    idx=np.random.choice(np.array(range(0,len(df_result_fin))), M)
    for i in range(len(idx)):
        random_point = Point(np.array(df_result_fin.iloc[idx]['coord_cent'])[i][0],
                             np.array(df_result_fin.iloc[idx]['coord_cent'])[i][1])
        sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

def generate_candidate_sites(df_result_fin,Weight,M=100):
    sites = []
    idx = df_result_fin.sort_values(by = Weight, ascending = False).iloc[1:M].index
    for i in range(len(idx)):
        random_point = Point(np.array(df_result_fin.loc[idx]['coord_cent'])[i][0],
                             np.array(df_result_fin.loc[idx]['coord_cent'])[i][1])
        sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

from scipy.spatial import distance_matrix
def mclp(points,K,radius,M,df_result_fin,w,Weight):
    """
    Solve maximum covering location problem
    Input:
        points: input points, Numpy array in shape of [N,2]
        K: the number of sites to select
        radius: the radius of circle
        M: the number of candidate sites, which will randomly generated inside
        the ConvexHull wrapped by the polygon
    Return:
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        f: the optimal value of the objective function
    """
    print('----- Configurations -----')
    print('  Number of points %g' % points.shape[0])
    print('  K %g' % K)
    print('  Radius %g' % radius)
    print('  M %g' % M)
    import time
    start = time.time()
    sites = generate_candidate_sites(df_result_fin,Weight,M)
    J = sites.shape[0]
    I = points.shape[0]
    D = distance_matrix(points,sites)
    mask1 = D<=radius
    D[mask1]=1
    D[~mask1]=0
    from mip import Model, xsum, maximize, BINARY
    # Build model
    m = Model("mclp")
    # Add variables
    x = [m.add_var(name = "x%d" % j, var_type = BINARY) for j in range(J)]
    y = [m.add_var(name = "y%d" % i, var_type = BINARY) for i in range(I)]
    m.objective = maximize(xsum(w[i]*y[i] for i in range (I)))
    m += xsum(x[j] for j in range(J)) == K
    for i in range(I):
        m += xsum(x[j] for j in np.where(D[i]==1)[0]) >= y[i]
    m.optimize()
    end = time.time()
    print('----- Output -----')
    print('  Running time : %s seconds' % float(end-start))
    print('  Optimal coverage points: %g' % m.objective_value)
    solution = []
    for i in range(J):
        if x[i].x ==1:
            solution.append(int(x[i].name[1:]))
    opt_sites = sites[solution]
    return opt_sites,m.objective_value

import pathlib
import random
from functools import reduce
from collections import defaultdict
import pandas as pd
import geopandas as gpd
import folium
import shapely
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
#import xgboost
import sklearn.cluster
#from geoband import API
import pydeck as pdk
import os
import pandas as pd
#import deckgljupyter.Layer as deckgl
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Nanum Gothic'
import numpy as np
from shapely.geometry import Polygon, Point
from numpy import random
#최적화 solver
import time
from mip import Model, xsum, maximize, BINARY

# df_result_fin = df_result[(df_result['개발가능']==1)]
# points = []
# for i in df_result_fin['coord_cent'] :
#     points.append([float(i[1 : -2].split(',')[0]), float(i[1 : -2].split(',')[1])])
# w= []
# for i in df_result_fin['w_FS'] :
#     w.append(i)
# radius = 100
# K = 5
# M = 5000
# opt_sites_org,f = mclp(np.array(points),K,radius,M,df_result_fin,w,'w_FS')
# df_opt_FS= pd.DataFrame(opt_sites_org)
# df_opt_FS.columns = ['lon', 'lat']
# df_opt_FS

input = st.text_input(label="자동차 대수", value="5", max_chars=10, help='input message < 20')

selected_item = st.radio("운행 시간", ("8시 - 13시", "13시 - 18시", "18시 - 23시"))

df_opt_FS = pd.read_csv('df_opt_FS', index_col = 0)
df_opt_SS = pd.read_csv('df_opt_SS', index_col = 0)
df_opt_LS = pd.read_csv('df_opt_LS', index_col = 0)

df_opt_FSS = df_opt_FS[0 : int(input)]
df_opt_SSS = df_opt_SS[0 : int(input)]
df_opt_LSS = df_opt_LS[0 : int(input)]

center = [126.5, 33.5]
if selected_item == "8시 - 13시":
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            longitude=center[0],
            latitude=center[1],
            zoom=10
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_opt_FSS,
                get_position='[lon, lat]',
                get_fill_color='[255, 255, 0]',
                get_radius=5000,
            )
        ],
    ))
elif selected_item == "13시 - 18시":
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            longitude=center[0],
            latitude=center[1],
            zoom=10
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_opt_SSS,
                get_position='[lon, lat]',
                get_fill_color='[255, 0, 100]',
                get_radius=5000,
            )
        ],
    ))
elif selected_item == "18시 - 23시":
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            longitude=center[0],
            latitude=center[1],
            zoom=10
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_opt_LSS,
                get_position='[lon, lat]',
                get_fill_color='[0, 0, 255]',
                get_radius=5000,
            )
        ],
    ))

from geopy.distance import great_circle
import pandas as pd
# import folium

class CountByWGS84:

    def __init__(self, df, lat, lon, dist=1):
        """
        df: 데이터 프레임
        lat: 중심 위도
        lon: 중심 경도
        dist: 기준 거리(km)
        """
        self.df = df
        self.lat = lat
        self.lon = lon
        self.dist = dist

    def filter_by_rectangle(self):
        """
        사각 범위 내 데이터 필터링
        """
        lat_min = self.lat - 0.01 * self.dist
        lat_max = self.lat + 0.01 * self.dist

        lon_min = self.lon - 0.015 * self.dist
        lon_max = self.lon + 0.015 * self.dist

        self.points = [[lat_min, lon_min], [lat_max, lon_max]]

        result = self.df.loc[
            (self.df['lat'] > lat_min) &
            (self.df['lat'] < lat_max) &
            (self.df['lon'] > lon_min) &
            (self.df['lon'] < lon_max)
        ]
        result.index = range(len(result))

        return result

    def filter_by_radius(self):
        """
        반경 범위 내 데이터 필터링
        """
        # 사각 범위 내 데이터 필터링
        tmp = self.filter_by_rectangle()

        # 기준 좌표 포인트
        center = (self.lat, self.lon)

        result = pd.DataFrame()
        for index, row in tmp.iterrows():
            # 개별 좌표 포인트
            point = (row['lat'], row['lon'])
            d = great_circle(center, point).kilometers
            if d <= self.dist:
                result = pd.concat([result, tmp.iloc[index, :].to_frame().T])

        result.index = range(len(result))

        return result

a = []
b = []
df_r = df_result[(df_result['개발가능']==1)]
df_r = df_r.reset_index()
for i in range(len(df_r)) :
  a.append(df_r['coord_cent'][i][0])
  b.append(df_r['coord_cent'][i][1])

df_r['lon'] = a
df_r['lat'] = b

lat = df_opt_SS['lat'][0]
lon = df_opt_SS['lon'][0]
dist = 5 #반경 1km설정
cbw = CountByWGS84(df_r, lat, lon, dist)

df_result1 = cbw.filter_by_radius()

print(f"""
{"="*50}
중심 위도: {cbw.lat}
중심 경도: {cbw.lon}
기준 거리: {cbw.dist} km
반경 범위 내 데이터 필터링 결과: {len(df_result1):,} 건
{"="*50}
""")

import streamlit as st
from streamlit_folium import folium_static
import folium

"# streamlit-folium"

with st.echo():
    import streamlit as st
    from streamlit_folium import folium_static
    import folium

    # center on Liberty Bell
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)

    # add marker for Liberty Bell
    tooltip = "Liberty Bell"
    folium.Marker(
        [39.949610, -75.150282], popup="Liberty Bell", tooltip=tooltip
    ).add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)