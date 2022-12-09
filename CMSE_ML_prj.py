import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pydeck as pdk


# -----Title of the dashborad
st.title('World Cities - Cost of Living')

tab1, tab2, tab3, tab4 = st.tabs(["1.Intro", "2.Methodology", "3.Unsupervised Learning","4.Geolocation & Findings"])

# -----Description of the project
tab1.subheader("1.1. Introduction")
tab1.markdown('The goal of this project is to apply unsupervised learning to cluster the world cities based on their living costs. With the optimal number of clusters discovered in this web application, we will label each city and explore the characteristics of each cluster. Also, we will locate each cluster on the world map with different colors to observe whether the results of clustering are related to geographic location.')
tab1.markdown('---')

tab1.subheader("1.2. Dataset")
tab1.markdown('The original data set consists of country names, city names, and categories for cost of living. I added continent names, latitude, and longitude on the data set to locate each of city on the world map.')
# -----Read CSV or Excel file and load data
#@st.cache
def load_data():
    # Read Excel
    excel_file= 'worldcities_cost_of_living.xlsx'
    sheet_name='Sheet1'
    df = pd.read_excel(excel_file,sheet_name=sheet_name,usecols='A:BH',header=0)
    return df

data=load_data()
data_ml=load_data()

# -----Define select variables
num_columns = data[['Meal, Inexpensive Restaurant','Meal for 2 People, Mid-range Restaurant, Three-course','McMeal at McDonalds (or Equivalent Combo Meal)','Domestic Beer (500ml draught)','Imported Beer (500ml bottle)','Coke/Pepsi (330ml bottle)','Water (330ml bottle)','Milk (regular), (1 liter)','Loaf of Fresh White Bread (500g)','Eggs (regular) (12)','Local Cheese (1kg)','Water (1500ml bottle)','Bottle of Wine (Mid-Range)','Domestic Beer (500ml bottle)','Imported Beer (330ml bottle)','Cigarettes 20 Pack (Marlboro)','One-way Ticket (Local Transport)','Chicken Breasts (Boneless, Skinless), (1kg)','Monthly Pass (Regular Price)','Gasoline (1 liter)','Volkswagen Golf','Apartment (1 bedroom) in City Centre','Apartment (1 bedroom) Outside of Centre','Apartment (3 bedrooms) in City Centre','Apartment (3 bedrooms) Outside of Centre','Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment','1 min of Prepaid Mobile Tariff Local (No Discounts or Plans)','Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)','Fitness Club, Monthly Fee for 1 Adult','Tennis Court Rent (1 Hour on Weekend)','Cinema, International Release, 1 Seat','1 Pair of Jeans (Levis 501 Or Similar)','1 Summer Dress in a Chain Store (Zara, H&M)','1 Pair of Nike Running Shoes (Mid-Range)','1 Pair of Men Leather Business Shoes','Price per Square Meter to Buy Apartment in City Centre','Price per Square Meter to Buy Apartment Outside of Centre','Average Monthly Net Salary (After Tax)','Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate','Taxi Start (Normal Tariff)','Taxi 1km (Normal Tariff)','Taxi 1hour Waiting (Normal Tariff)','Apples (1kg)','Oranges (1kg)','Potato (1kg)','Lettuce (1 head)','Cappuccino (regular)','Rice (white), (1kg)','Tomato (1kg)','Banana (1kg)','Onion (1kg)','Beef Round (1kg) (or Equivalent Back Leg Red Meat)','Toyota Corolla Comfort (Or Equivalent New Car)','Preschool (or Kindergarten), Full Day, Private, Monthly for 1 Child','International Primary School, Yearly for 1 Child']].columns 
loc_columns = data[['City','Country','Continent']].columns
num_columns_ml = data_ml[['Meal, Inexpensive Restaurant','Meal for 2 People, Mid-range Restaurant, Three-course','McMeal at McDonalds (or Equivalent Combo Meal)','Domestic Beer (500ml draught)','Imported Beer (500ml bottle)','Coke/Pepsi (330ml bottle)','Water (330ml bottle)','Milk (regular), (1 liter)','Loaf of Fresh White Bread (500g)','Eggs (regular) (12)','Local Cheese (1kg)','Water (1500ml bottle)','Bottle of Wine (Mid-Range)','Domestic Beer (500ml bottle)','Imported Beer (330ml bottle)','Cigarettes 20 Pack (Marlboro)','One-way Ticket (Local Transport)','Chicken Breasts (Boneless, Skinless), (1kg)','Monthly Pass (Regular Price)','Gasoline (1 liter)','Volkswagen Golf','Apartment (1 bedroom) in City Centre','Apartment (1 bedroom) Outside of Centre','Apartment (3 bedrooms) in City Centre','Apartment (3 bedrooms) Outside of Centre','Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment','1 min of Prepaid Mobile Tariff Local (No Discounts or Plans)','Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)','Fitness Club, Monthly Fee for 1 Adult','Tennis Court Rent (1 Hour on Weekend)','Cinema, International Release, 1 Seat','1 Pair of Jeans (Levis 501 Or Similar)','1 Summer Dress in a Chain Store (Zara, H&M)','1 Pair of Nike Running Shoes (Mid-Range)','1 Pair of Men Leather Business Shoes','Price per Square Meter to Buy Apartment in City Centre','Price per Square Meter to Buy Apartment Outside of Centre','Average Monthly Net Salary (After Tax)','Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate','Taxi Start (Normal Tariff)','Taxi 1km (Normal Tariff)','Taxi 1hour Waiting (Normal Tariff)','Apples (1kg)','Oranges (1kg)','Potato (1kg)','Lettuce (1 head)','Cappuccino (regular)','Rice (white), (1kg)','Tomato (1kg)','Banana (1kg)','Onion (1kg)','Beef Round (1kg) (or Equivalent Back Leg Red Meat)','Toyota Corolla Comfort (Or Equivalent New Car)','Preschool (or Kindergarten), Full Day, Private, Monthly for 1 Child','International Primary School, Yearly for 1 Child']].columns 
loc_columns_ml = data_ml[['City','Country','Continent']].columns

# loc_columns = data[['City','Country','Continent','Latitude','Longitude']].columns

# ------checkbox
checkbox_1 = tab1.checkbox("Reveal the dataset")

# -----Column Description
expander = tab1.expander("See the columns")
expander.write("""City""")
expander.write("""Country""")
expander.write("""Continent""")
expander.write("""Latitude""")
expander.write("""Longitude""")
expander.write("""Meal, Inexpensive Restaurant""")
expander.write("""Meal for 2 People, Mid-range Restaurant, Three-course""")
expander.write("""McMeal at McDonalds (or Equivalent Combo Meal)""")
expander.write("""Domestic Beer (0.5 liter draught)""")
expander.write("""Imported Beer (0.5 liter bottle)""")
expander.write("""Coke/Pepsi (0.33 liter bottle)""")
expander.write("""Water (0.33 liter bottle)""")
expander.write("""Milk (regular), (1 liter)""")
expander.write("""Loaf of Fresh White Bread (500g)""")
expander.write("""Eggs (regular) (12)""")
expander.write("""Local Cheese (1kg)""")
expander.write("""Water (1.5 liter bottle)""")
expander.write("""Bottle of Wine (Mid-Range)""")
expander.write("""Domestic Beer (0.5 liter bottle)""")
expander.write("""Imported Beer (0.33 liter bottle)""")
expander.write("""Cigarettes 20 Pack (Marlboro)""")
expander.write("""One-way Ticket (Local Transport)""")
expander.write("""Chicken Breasts (Boneless, Skinless), (1kg)""")
expander.write("""Monthly Pass (Regular Price)""")
expander.write("""Gasoline (1 liter)""")
expander.write("""Volkswagen Golf""")
expander.write("""Apartment (1 bedroom) in City Centre""")
expander.write("""Apartment (1 bedroom) Outside of Centre""")
expander.write("""Apartment (3 bedrooms) in City Centre""")
expander.write("""Apartment (3 bedrooms) Outside of Centre""")
expander.write("""Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment""")
expander.write("""1 min. of Prepaid Mobile Tariff Local (No Discounts or Plans)""")
expander.write("""Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)""")
expander.write("""Fitness Club, Monthly Fee for 1 Adult""")
expander.write("""Tennis Court Rent (1 Hour on Weekend)""")
expander.write("""Cinema, International Release, 1 Seat""")
expander.write("""1 Pair of Jeans (Levis 501 Or Similar)""")
expander.write("""1 Summer Dress in a Chain Store (Zara, H&M)""")
expander.write("""1 Pair of Nike Running Shoes (Mid-Range)""")
expander.write("""1 Pair of Men Leather Business Shoes""")
expander.write("""Price per Square Meter to Buy Apartment in City Centre""")
expander.write("""Price per Square Meter to Buy Apartment Outside of Centre""")
expander.write("""Average Monthly Net Salary (After Tax)""")
expander.write("""Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate""")
expander.write("""Taxi Start (Normal Tariff)""")
expander.write("""Taxi 1km (Normal Tariff)""")
expander.write("""Taxi 1hour Waiting (Normal Tariff)""")
expander.write("""Apples (1kg)""")
expander.write("""Oranges (1kg)""")
expander.write("""Potato (1kg)""")
expander.write("""Lettuce (1 head)""")
expander.write("""Cappuccino (regular)""")
expander.write("""Rice (white), (1kg)""")
expander.write("""Tomato (1kg)""")
expander.write("""Banana (1kg)""")
expander.write("""Onion (1kg)""")
expander.write("""Beef Round (1kg) (or Equivalent Back Leg Red Meat)""")
expander.write("""Toyota Corolla Comfort (Or Equivalent New Car)""")
expander.write("""Preschool (or Kindergarten), Full Day, Private, Monthly for 1 Child""")
expander.write("""International Primary School, Yearly for 1 Child""")

if checkbox_1:
    tab1.dataframe(data=data)


# ----- 2.Methodology
tab2.markdown('As you see in the previous tab, there are more than 60 columns in the dataset. In the real world, it is impossible to make a 60-dimensional graph. To understand the K-mean clustering method which we are going to explore in this web app, we will use two columns instead to visualize them on a 2-dimensional graph.')
tab2.markdown("""Let's see how we get the optimal number of clusters and what the clustering result looks like!""")
tab2.markdown('---')

# ----- 2.1.
tab2.subheader('2.1. Scatter Plot')
tab2.caption('Pick any two columns you want! You can also color the scattered dots if you want.')
col1, col2, col3 = tab2.columns(3)
with col1:
    select_box1 = st.selectbox(label = "X-axis", options = num_columns)
with col2:
    select_box2 = st.selectbox(label = "Y-axis", options = num_columns)
with col3:
    select_box3 = st.selectbox(label = "Hue", options = loc_columns)
sct_chart_1 = alt.Chart(data).mark_circle().encode( # remove color
    x=select_box1, y=select_box2, color=select_box3, tooltip=['City','Country','Continent'])
tab2.altair_chart(sct_chart_1)
tab2.markdown('Can you identify any clusters in the graph above?')
tab2.markdown('---')

# ----- 2.2.
tab2.subheader('2.2. Optimal Number of Clusters')
tab2.markdown("""I've tried the column 'Meal, Inexpensive Restaurant' on the X-axis and 'Bottle of Wine (Mid-Range)' on the Y-axis. And, to be honest, I cannot see any good clusters on the graph. Then, how do we know how many clusters we need for unsupervised learning? The answer is...""")
tab2.markdown('Take a look at these elbow curve and silhouette score!')

col1, col2 = tab2.columns(2)
with col1:
    # Elbow Curve
    st.markdown('Elbow Curve:')
    distance = []
    for k in range(2,10):
        k_model = KMeans(n_clusters=k)
        k_model.fit(data[[select_box1,select_box2]])
        distance.append(k_model.inertia_)

    fig_1, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(x=range(2,10), y=distance)
    st.write(fig_1)
    st.caption('The elbow indicates the optimal k.')

with col2:
    # Silhouette Score
    st.markdown('Silhouette Score:')
    silhouette = []
    for k in range(2,10):
        k_model = KMeans(n_clusters=k)
        k_model.fit(data[[select_box1,select_box2]])
        labels = k_model.predict(data[[select_box1,select_box2]])
        silhouette.append(silhouette_score(data[[select_box1,select_box2]], labels))

    fig_2, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(x=range(2,10),y=silhouette)
    st.write(fig_2)
    st.caption('The maximum silhouette score indicates the optimal k.')
tab2.markdown('---')

# ----- 2.3.
tab2.subheader('2.3. K-means Clustering Result')
tab2.markdown('We checked that the optimal number of clusters is 3. When we set the slider to 3, the data points are categorized into the three clusters which share certain similarities.')
cluster = tab2.slider('If you want to change the number of clusters, adjust the slider below.', 2, 10, 3)
kmeans_model = KMeans(n_clusters=cluster, random_state=100)
kmeans_model.fit(data[[select_box1,select_box2]])
data['label'] = kmeans_model.predict(data[[select_box1,select_box2]])
sct_chart_2 = alt.Chart(data).mark_circle().encode(
    x=select_box1, y=select_box2, color='label', tooltip=['label','City','Country','Continent'])
tab2.altair_chart(sct_chart_2)



# ----- 3.Unsupervised Learning
tab3.markdown("""There is no scatter plot this time because we are going to use the whole dataset with 60 columns. But, don't worry! We will follow the exactly same steps what we've just did in the '2.Methodology' tab.""")
tab3.markdown('---')

# ----- 3.1.
tab3.subheader("3.1.Optimal Number of Clusters")
col1, col2 = tab3.columns(2)

with col1:
    # Elbow Curve
    st.markdown('Elbow Curve:')
    distance_ml = []
    for k in range(2,10):
        k_model_ml = KMeans(n_clusters=k)
        k_model_ml.fit(data_ml[num_columns_ml])
        distance_ml.append(k_model_ml.inertia_)

    fig_3, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(x=range(2,10), y=distance_ml)
    st.write(fig_3)

with col2:
    # Silhouette Score
    st.markdown('Silhouette Score:')
    silhouette_ml = []
    for k in range(2,10):
        k_model_ml = KMeans(n_clusters=k)
        k_model_ml.fit(data_ml[num_columns_ml])
        labels_ml = k_model_ml.predict(data_ml[num_columns_ml])
        silhouette_ml.append(silhouette_score(data[num_columns_ml], labels_ml))

    fig_4, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(x=range(2,10),y=silhouette_ml)
    st.write(fig_4)
tab3.markdown('The optimal number of clusters is 4.')
tab3.markdown('---')

# ----- 3.2.
tab3.subheader('3.2. K-means Clustering Result')
cluster_ml = tab3.slider('If you want to change the number of clusters, try the slider below.', 2, 10, 4)

# K-means Clustring output
kmeans_model_ml = KMeans(n_clusters=cluster_ml, random_state=100)
kmeans_model_ml.fit(data_ml[num_columns_ml])
kmeans_model_ml.predict(data_ml[num_columns_ml])
data_ml['label'] = kmeans_model_ml.predict(data_ml[num_columns_ml])

# Number of City Per Cluster Table
data_ml_count = data_ml.groupby('label').count()['Meal, Inexpensive Restaurant']
data_ml_count = data_ml_count.rename('Number of City Per Cluster') #rename
data_ml_count_df = data_ml_count.to_frame()
data_ml_count_df_t = data_ml_count_df.T
tab3.write(data_ml_count_df_t)
tab3.markdown("""We can generate four clusters when k equals 4. Each cluster's name is 0, 1, 2, and 3. The interesting fact I have found from the table above is that the cluster label 2 has only one data point. It turned out to be Singapore, which means that Singapore has unique characteristics different from the other three clusters.""")
# tab3.markdown('---')

# Bar graph
tab3.markdown("""Let's find out the characteristics of each cluster.""")
select_box3 = tab3.selectbox(label = 'Cost of Living Category:', options = num_columns_ml)
fig_t, ax = plt.subplots()
df_groups = data_ml.groupby('label')[select_box3].mean()
df_cluster_mean = df_groups.mean() # cluster mean
df_total_mean = data_ml[select_box3].mean() # total mean
ax = plt.axhline(y = df_cluster_mean, color = 'b', linestyle = 'dashed', label = "Mean of the clusters")  # cluster mean
ax = plt.axhline(y = df_total_mean, color = 'r', linestyle = 'dashed', label = "Mean of the total cities")  # total mean
ax = plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
ax = df_groups.plot(kind='bar')
ax = plt.xlabel('Cluster Label')
tab3.pyplot(fig_t)

# Check box
data_ml_mean = data_ml.groupby('label').mean()
data_ml_mean_groupby = data_ml_mean.drop(['Latitude','Longitude'],axis=1)
data_ml_mean_groupby_t = data_ml_mean_groupby.T
# Calculate overall mean
data_ml_total_mean = data_ml.drop(['Latitude','Longitude'],axis=1).mean()
data_ml_total_mean = data_ml_total_mean.rename('mean')
# Join overall mean
# data_ml_join_mean = data_ml_groupby_t.join(data_ml_total_mean)

checkbox_2 = tab3.checkbox("Reveal the summary of result")
if checkbox_2:
    # tab3.write(data_ml_count_df_t)
    tab3.write(data_ml_mean_groupby_t)

checkbox_3 = tab3.checkbox("See the cluster's details")
if checkbox_3:
    select_box_3 = tab3.selectbox(label = "select the label for details:", options = range(cluster_ml))
    data_ml_by_label = data_ml[data_ml['label']==select_box_3].drop(['Latitude','Longitude'],axis=1)
    tab3.write(data_ml_by_label)



# ----- 4.Geolocation and Findings
# ----- 4.1.
tab4.subheader('4.1.Geolocation')
geo_data_0 = data_ml[data_ml["label"]==0].iloc[:,3:5]
geo_data_1 = data_ml[data_ml["label"]==1].iloc[:,3:5]
geo_data_2 = data_ml[data_ml["label"]==2].iloc[:,3:5]
geo_data_3 = data_ml[data_ml["label"]==3].iloc[:,3:5]
geo_data_4 = data_ml[data_ml["label"]==4].iloc[:,3:5]
geo_data_5 = data_ml[data_ml["label"]==5].iloc[:,3:5]
geo_data_6 = data_ml[data_ml["label"]==6].iloc[:,3:5]
geo_data_7 = data_ml[data_ml["label"]==7].iloc[:,3:5]
geo_data_8 = data_ml[data_ml["label"]==8].iloc[:,3:5]
geo_data_9 = data_ml[data_ml["label"]==9].iloc[:,3:5]

tab4.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=44.978718,
        longitude=-84.515887,
        zoom=1,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_0,
            get_position='[Longitude, Latitude]',
            get_color='[240, 231, 104, 160]', #yellow
            get_radius=200000,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_1, 
            get_position='[Longitude, Latitude]',
            get_color='[132, 169, 255, 160]', #blue
            get_radius=200000,
        ),     
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_2, 
            get_position='[Longitude, Latitude]',
            get_color='[111, 215, 121, 160]', #green
            get_radius=200000,
        ),  
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_3, 
            get_position='[Longitude, Latitude]',
            get_color='[255, 179, 102, 160]', #orange
            get_radius=200000,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_4, 
            get_position='[Longitude, Latitude]',
            get_color='[192, 143, 247, 160]', #purple
            get_radius=200000,
        ), 
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_5, 
            get_position='[Longitude, Latitude]',
            get_color='[255, 131, 131, 160]', #red
            get_radius=200000,
        ), 
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_6, 
            get_position='[Longitude, Latitude]',
            get_color='[164, 107, 34, 160]', #brown
            get_radius=200000,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_7, 
            get_position='[Longitude, Latitude]',
            get_color='[178, 175, 171, 160]', #gray
            get_radius=200000,
        ), 
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_8, 
            get_position='[Longitude, Latitude]',
            get_color='[255, 210, 248, 160]', #pink
            get_radius=200000,
        ), 
        pdk.Layer(
            'ScatterplotLayer',
            data=geo_data_9, 
            get_position='[Longitude, Latitude]',
            get_color='[92, 49, 5, 160]', #dark brown
            get_radius=200000,
        ),                                                                                             
    ],
))

tab4.markdown('Label 0: Yellow | Label 1: Blue | Label 2: Green | Label 3: Orange | Label 4: Purple | Label 5: Red | Label 6: Brown | Label 7: Gray | Label 8: Pink | Label 9: Dark brown')
tab4.markdown('---')

# ----- 4.2.
tab4.subheader('4.2.Findings')

tab4.markdown('**[Cluster: Label 0]**')
tab4.markdown("""* **The cluster of the absolute low living cost if you don't have a mortgage loan.**""")
tab4.markdown("""* The average cost of living in this cluster is the lowest in all categories except the category 'Mortgage Interest Rate.'""")
tab4.markdown('* The Mortgage Interest Rate is the highest among the four clusters.')
# tab4.markdown('* Countries and cities')

tab4.markdown('**[Cluster: Label 1]**')
tab4.markdown('* **The cluster of the relative low living cost.**')
tab4.markdown("""* The average cost of living in this cluster is similar to the cluster 'label 3'""")
tab4.markdown("""* However, the 'Average Monthly Net Salary' is twice as much as the cluster label 3.""")
# tab4.markdown('* Countries and cities')

tab4.markdown('**[Cluster: Label 2]**')
tab4.markdown('* **The cluster of the absolute high living cost.**')
tab4.markdown('* The most expensive city in the world to buy a car. Cars in Singapore cost on approximately 4 times more than they do in the other clusters.')
tab4.markdown('* Also, the average cost of living in most of categories are higher than average.')

tab4.markdown('**[Cluster: Label 3]**')
tab4.markdown('* **The cluster of the relative high living cost.**')
tab4.markdown("""* The average cost of living in this cluster is similar to the cluster label 1""")
tab4.markdown('* However, the Average Monthly Net Salary is twice as small as the cluster label 1.')
