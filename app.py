import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import silhouette_score
import io
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO

# Setting random seed for reproducibility
seed = 42
np.random.seed(seed)
st.set_page_config(layout="wide")

st.title("Kena RFM Segmentation and Clustering")



# File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    # Data preview
    with st.expander("## 🔎 Data Preview"):
        st.dataframe(df)
    st.markdown("## 📝 RFM Calculation Details")
    st.markdown("""
        The RFM model segments customers based on these three factors, providing valuable insights into customer behavior and helping to identify key customer groups for targeted marketing strategies.
                
        **Recency, Frequency, and Monetary Value (RFM) Calculation:**

        - **Recency**: This metric represents the number of days from payment gateway launch to the patient's last purchase. It is calculated by taking the difference between the last consultation date for each patient and the launch date 2022-08-23. Thus the most recent consultaitons will have the highest recency. 

        - **Frequency**: This metric indicates how often a patient consults since paygate. It is calculated as the total number of invoices generated against each patient.

        - **Monetary Value**: This metric represents the total amount of money invoiced to the patient. It is calculated by summing up the monetary value of all invoices generated against each patient.
        
    """)
    st.markdown("## 🛠️ Preprocessing")
    # Convert date column to datetime format
    df['Date'] = pd.to_datetime(df['CREATED_AT'])
    df['Rank'] = df.sort_values(['PATIENT_ID', 'Date']).groupby(['PATIENT_ID'])['Date'].rank(method='min').astype(int)
    df_rec = df[df['Rank'] == 1]

    df_rec['Recency'] = (df_rec['Date'] - pd.to_datetime(min(df_rec['Date']))).dt.days
    freq = df.groupby('PATIENT_ID').size()
    df_freq = pd.DataFrame(freq, columns=['Frequency']).reset_index()
    rec_freq = df_freq.merge(df_rec, on='PATIENT_ID')

    m = df.groupby('PATIENT_ID')['AMOUNT'].sum()
    m = pd.DataFrame(m).reset_index()
    m.columns = ['PATIENT_ID', 'Monetary_value']

    rfm = m.merge(rec_freq, on='PATIENT_ID')
    finaldf = rfm[['PATIENT_ID', 'Recency', 'Frequency', 'Monetary_value']]

    list1 = ['Recency', 'Frequency', 'Monetary_value']
    # Apply a Seaborn style
    sns.set_style("whitegrid")

    # Create columns for side-by-side plots
    cols = st.columns(len(list1))

    # Iterate over each feature to create a boxplot
    for i, col in zip(list1, cols):
        with col:
            st.write(f"### {i} Distribution")
            
            # Set up the figure size and style
            plt.figure(figsize=(6, 4))
            ax = sns.boxplot(x=finaldf[str(i)], palette="Set2")

            # Add a title and labels
            plt.title(f'{i} Boxplot', fontsize=14)
            plt.xlabel(i, fontsize=12)

            # Display the plot in Streamlit
            st.pyplot(plt.gcf())
            
            # Clear the current figure to avoid overlap
            plt.clf()
    
    st.write("""
            ## 🧠 Understanding the Z-Score and Outlier Removal

            The z-score is a statistical measure that indicates how many standard deviations a data point is from the mean of the dataset. The formula for calculating the z-score is:

            $$ Z = \\frac{(X - \\mu)}{\\sigma} $$

            Where:
            - **X** is the value of the data point,
            - **μ** (mu) is the mean of the dataset,
            - **σ** (sigma) is the standard deviation of the dataset.

            The z-score helps identify outliers, which are data points that are significantly different from others in the dataset. In this case, outliers are defined as points with a z-score greater than 3 or less than -3, meaning they fall outside 3 standard deviations from the mean.

            To learn more about the z-score, you can visit [this article](https://en.wikipedia.org/wiki/Standard_score).
            """)

    new_df = finaldf[['Recency', 'Frequency', 'Monetary_value']]
    z_scores = stats.zscore(new_df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = new_df[filtered_entries]

    # Display DataFrame info side by side before and after outlier removal
    col1, col2 = st.columns(2)

    with col1:
        st.write("## DataFrame Info Before Outlier Removal")
        buffer = io.StringIO()
        finaldf.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    with col2:
        st.write("## DataFrame Info After Outlier Removal")
        buffer = io.StringIO()
        new_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    st.markdown("---")  # Section separator

    st.write("## K-Means Clustering")

    col_names = ['Recency', 'Frequency', 'Monetary_value']
    features = new_df[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features = pd.DataFrame(features, columns=col_names)

    SSE = []
    for cluster in range(1, 10):
        kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state=seed)
        kmeans.fit(scaled_features)
        SSE.append(kmeans.inertia_)

    frame = pd.DataFrame({'Cluster': range(1, 10), 'SSE': SSE})

    plt.figure(figsize=(8, 4))

    # Plot the SSE values
    plt.plot(frame['Cluster'], frame['SSE'], marker='o', color='b', linestyle='-', linewidth=2, markersize=8)

    # Add title and labels with enhanced font sizes
    plt.title('Elbow Method For Optimal Number of Clusters', fontsize=18)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Sum of Squared Distances (Inertia)', fontsize=14)

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)

    # Add x-ticks and y-ticks with better font sizes
    plt.xticks(frame['Cluster'], fontsize=12)
    plt.yticks(fontsize=12)

    # Add a background color to the plot area
    plt.gca().set_facecolor('#f9f9f9')

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())

    # Clear the plot to avoid overlap
    plt.clf()

    # User input for number of clusters
    with st.container():
        st.markdown("### 🔢 **Cluster Selection**")
        
        st.markdown("""
        **Please select the number of clusters:**  
        Adjust the drop down below to choose how many clusters you want for the KMeans algorithm. 
        The default value is set to 5.
        """)

        # Highlighted slider for selecting the number of clusters
        num_clusters = st.selectbox("Select number of clusters", options=list(range(2, 11)), index=3)


        st.markdown("""
        *This choice directly influences the clustering outcome, so consider experimenting with different values to see how they affect the segmentation.*
        """)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=seed)
    kmeans.fit(scaled_features)

    silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
    with st.container():
        st.markdown("### 🌟 **Silhouette Score**")

        st.markdown("""
        <div style="font-size: 18px; text-align: center;">
            **Silhouette Score:** <span style="color: #ff0000; font-weight: bold; font-size: 24px;">{:.4f}</span>
        </div>
        """.format(silhouette_avg), unsafe_allow_html=True)


        st.markdown("""
        **What is the Silhouette Score?**

        The Silhouette Score is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). It ranges from -1 to 1:
        
        - **+1** indicates that the sample is far away from the neighboring clusters and very close to the cluster it belongs to.
        - **0** indicates that the sample is on or very close to the decision boundary between two neighboring clusters.
        - **-1** indicates that the sample might have been assigned to the wrong cluster.
                    
        *A higher silhouette score suggests that the chosen number of clusters is appropriate and that the clustering is effective.*
        """)

    pred = kmeans.predict(scaled_features)
    frame = pd.DataFrame(new_df)
    frame['cluster'] = pred

    avg_df = frame.groupby(['cluster'], as_index=False).mean()

    palette = sns.color_palette("tab10", n_colors=avg_df['cluster'].nunique())
    color_mapping = {cluster: palette[i] for i, cluster in enumerate(sorted(avg_df['cluster'].unique()))}

    finaldf_with_clusters = pd.merge(finaldf, frame[['cluster']], left_index=True, right_index=True, how='inner')
    cluster_counts = finaldf_with_clusters['cluster'].value_counts()

    

    st.markdown("### 📊 **Clustering Results**")
    # Plotting the clusters side by side with consistent hue and displaying values
    cols = st.columns(len(list1))

    for i, col in zip(list1, cols):
        with col:
            plt.figure(figsize=(6, 4))
            sns.barplot(x='cluster', y=str(i), data=avg_df, palette=palette)
            plt.title(f'Average {i} by Cluster', fontsize=14)
            plt.xlabel('Cluster', fontsize=12)
            plt.ylabel(f'Average {i}', fontsize=12)

            # Display the values on the bars
            for index, value in enumerate(avg_df[str(i)]):
                plt.text(index, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

            st.pyplot(plt.gcf())
            plt.clf()
    
    with st.container():
    # Add an empty column on each side for centering
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio to make the middle column larger

        with col2:  # The middle column
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140,
                colors=[color_mapping[cluster] for cluster in cluster_counts.index], explode=[0.1] * len(cluster_counts),
                shadow=True, wedgeprops={'edgecolor': 'black', 'linewidth': 1})

            ax.set_title('Proportion of Clusters to Population', fontsize=14)
            st.pyplot(fig)
            plt.clf()
    st.markdown("## 🏷️ Customer Segment Descriptions")
    st.markdown("""
        ### Super Users:
        - **High Frequency** and **High Monetary Value**.
        - **Moderate to High Recency** (i.e., they consult frequently and have spent a lot recently).
        - Typically, these are our best users who consult regularly and spend the most.

        ### Loyal Users:
        - **Moderate to High Frequency**.
        - **Moderate Monetary Value**.
        - **High Recency**.
        - These users consult regularly but might not spend as much as Super Users.

        ### At-Risk Users:
        - **Low to Moderate Frequency**.
        - **Moderate to High Monetary Value**.
        - **Low Recency** (i.e., they haven’t consulted in a while).
        - These users used to spend well but haven’t had a consultation recently, so they might be at risk of churning.

        ### New Users:
        - **High Recency** (i.e., they have had a consultation recently).
        - **Low Frequency**.
        - **Low Monetary Value**.
        - These are users who have had a recent consultation but have not yet developed into loyal customers.

        ### Potential Loyal Users:
        - **Moderate Recency**.
        - **Moderate to High Frequency**.
        - **Moderate Monetary Value**.
        - These users are on their way to becoming loyal if nurtured properly.

        ### Churned Users:
        - **Low Recency**.
        - **Low Frequency**.
        - **Low Monetary Value**.
        - These are users who haven’t had a consultation in a long time and have a low engagement level.
        """)
    
    filtered_df = finaldf_with_clusters
    # filtered_df.to_excel('RFM_Segments_clusters_0_and_1.xlsx', index=False)
    # st.success("Clusters 0 and 1 have been exported to 'RFM_Segments_clusters_0_and_1.xlsx'")
    finaldf_with_clusters = pd.merge(finaldf_with_clusters, df[['PATIENT_ID', 'ID_NUMBER']], on='PATIENT_ID', how='left')

    # Preview the data under an expander
    with st.expander("Preview the Segmented Patients"):
        st.dataframe(finaldf_with_clusters)

    # Provide a download button for the Excel file
    buffer = BytesIO()
    finaldf_with_clusters.to_excel(buffer, index=False)
    buffer.seek(0)
    st.markdown(
        """
        <style>
        .stDownloadButton > button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            border: 2px solid #ff4b4b;
        }
        .stDownloadButton > button:hover {
            background-color: #ff0000;
            border: 2px solid #ff0000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.download_button(
        label="Download Data as Excel",
        data=buffer,
        file_name='RFM_Segments_clusters.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    # Optionally, visualize other details like age group or gender distribution by cluster
    # Add more plots and analysis as needed

else:
    st.warning("Please upload a CSV or Excel file.")
