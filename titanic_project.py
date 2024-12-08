import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset automatically from root directory if available
csv_file_path = "train.csv"


# Function to calculate and render missing values summary for a given DataFrame
def render_missing_values(df):
    # Step 1: Calculate the number of missing values per column
    missing_values = df.isnull().sum()
    
    # Filter out only the columns with missing values
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        st.write("No missing values in the dataset!")
    else:
        # Create a DataFrame with the missing values summary
        missing_values_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values Count': missing_values.values,
            'Percentage Missing': (missing_values.values / len(df)) * 100
        })

        # Optionally, display the summary as a static table
        st.markdown("### Missing Values Summary")
        st.write("Below is a summary of missing values in the dataset:")
        st.table(missing_values_df)

        # Render a heatmap to visualize missing data
        st.markdown("##### A heatmap that visualizes the missing values across the dataset (df)")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isna().transpose(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
        st.pyplot(plt.gcf())


if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Set up the title and introduction
    st.title("Exploratory Analysis of Titanic Dataset")
    st.markdown("### Project Description:")
    st.markdown("> This dataset was obtained from Kaggle. It focuses on the tragic event of the Titanic sinking. The dataset contains detailed information on the passengers aboard the RMS Titanic, including 12 features such as age, gender, passenger class, and survival status, with 891 instances in total.")

    st.markdown("> The data is preprocessed and explored to identify significant factors that influenced the survival of passengers by analyzing it through different exploratory techniques and visualizations. The insights gained from this analysis were used to develop an interactive and informative solution that provides an overview of the findings.")

    st.markdown("### Problem Statement:")
    st.markdown("> The sinking of the RMS Titanic is one of the most infamous disasters in history, but what factors determined who survived and who did not?")

    st.markdown("> Using data from Kaggle, this project aims to analyze the characteristics of the passengers to identify the key factors that influenced survival rates during the disaster. By applying data exploration and machine learning techniques, I seek to uncover insights and predict which kinds of passengers had the highest likelihood of survival.    This analysis will help us understand the circumstances that contributed to survival outcomes in this tragic event.") 


    ###### Sidebar author info section
    # Author profile photo with applied profile-photo class
    st.sidebar.image("profile.png", caption="", width=150, use_container_width =True, output_format="auto", clamp=False)

    st.sidebar.markdown(
        """
        <br>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Dataset Filter Options")

    # First select box in the sidebar: Filter by Passenger Class
    pclass = st.sidebar.selectbox("Select Passenger Class", options=["All"] + df['Pclass'].unique().tolist(), key='pclass_select')

    # Second select box in the sidebar: Filter by Sex
    sex = st.sidebar.selectbox("Select Sex", options=["All"] + df['Sex'].unique().tolist(), key='sex_select')

    # Third select box in the sidebar: Filter by Survival Status
    survived = st.sidebar.selectbox("Select Survival Status", options=["All"] + df['Survived'].unique().astype(str).tolist(), key='survived_select')

    # Sidebar multiselect to choose columns to display
    selected_columns = st.sidebar.multiselect(
        "Select Columns to Display", options=df.columns.tolist(), default=df.columns.tolist()
    ) 

    ### EDA Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Dataset info:")
    st.image("dataset_description.png", caption="", width=150, use_container_width =True, output_format="auto", clamp=False)


    # Apply filters to the DataFrame
    filtered_df = df.copy()

    # Apply filtering based on select box values if not 'All'
    if pclass != "All":
        filtered_df = filtered_df[filtered_df['Pclass'] == pclass]

    if sex != "All":
        filtered_df = filtered_df[filtered_df['Sex'] == sex]

    if survived != "All":
        # Convert 'Survived' select box value back to integer for filtering
        filtered_df = filtered_df[filtered_df['Survived'] == int(survived)]

    # Handle display of columns based on the multiselect
    if len(selected_columns) > 0:
        # Show only the selected columns in the filtered DataFrame
        filtered_df = filtered_df[selected_columns]

        # Display the filtered dataframe in the main area
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("The Titanic Dataset")
        st.write(filtered_df)
    else:
        # If no columns are selected, display the entire DataFrame (default view)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("The Titanic Dataset")
        st.write(df)
        st.warning("No columns selected. Displaying the entire dataset by default.")

    # Render the missing values summary using the defined function
    render_missing_values(df)
    st.info("Handling missing values is an important step in data preprocessing because missing data can skew analyses, lead to biases, or even render some algorithms unusable. There are several approaches to handling missing values such as:  Imputation (Filling Missing Values), Removing Missing Data (Removing Rows or Columns), Using Predictive Models (Predict the missing value based on other features.) e.t.c")


    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Visualizations for the Distribution of target Variable (Survival Status)")
    st.markdown("##### Purpose: Visualize the count of passengers in each class of the target variable (survived vs not survived).")
    plt.figure(figsize=(3, 3))
    sns.countplot(x='Survived', data=df, palette='pastel')
    plt.title('Count of Survival Status', fontsize=6)
    plt.xlabel('Survival Status (0 = Not Survived, 1 = Survived)', fontsize=6)
    plt.ylabel('Number of Passengers', fontsize=6)
    st.pyplot(plt.gcf())
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### Purpose: Visualize the distribution of a related variable like age, segmented by the target variable to observe the relationship between age and survival.")
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x='Age',
        hue='Survived',
        multiple='stack', 
        palette='Set2',
        binwidth=3,
        edgecolor='black'
    )

    plt.title('Age Distribution by Survival Status', fontsize=12)
    plt.xlabel('Age', fontsize=10)
    plt.ylabel('Count of Passengers', fontsize=10)
    plt.legend(title='Survived', labels=['0 - Not Survived', '1 - Survived'])

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### Purpose: Survival Rate by Passenger Class and Sex")
    if len(df) > 0: 
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Pclass', hue='Survived')
        st.pyplot(plt)
    else:
        st.write("No data available for the selected filters.")


    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### Purpose: Proportion of Survivors vs Non-Survivors.")
    survival_counts = df['Survived'].value_counts()
    plt.figure(figsize=(4, 4))
    plt.pie(
        survival_counts,
        labels=['Not Survived', 'Survived'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['lightcoral', 'skyblue']
    )
    plt.axis('equal') 
    st.pyplot(plt.gcf())

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### Purpose: Survival Rate by Passenger Class.")
    survival_rate_by_class = df.groupby('Pclass')['Survived'].mean()
    plt.figure(figsize=(5, 5))
    survival_rate_by_class.plot(kind='bar', color='skyblue')
    plt.title('Survival Rate by Passenger Class', fontsize=8)
    plt.xlabel('Passenger Class', fontsize=6)
    plt.ylabel('Average Survival Rate', fontsize=6)
    plt.xticks(rotation=0)
    st.pyplot(plt.gcf())
    st.info("This bar plot provides an intuitive way to observe the survival rates across different classes (Pclass). Seeing the average survival rates visually can make it easier to observe that first-class passengers were more likely to survive, while third-class passengers faced a much lower survival probability.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### Purpose: Pairplot of Survival and Fare.")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Fare', hue='Survived', fill=True, palette='Set1', alpha=0.4, linewidth=1.5)

    # Customize labels and title with reduced font size
    plt.title('Fare Distribution by Survival Status', fontsize=14)  # Smaller title
    plt.xlabel('Fare', fontsize=12)  # Smaller x-axis label
    plt.ylabel('Density', fontsize=12)  # Smaller y-axis label

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    st.info("From the KDE (Kernel Density Estimate) we can see that the red distribution (representing passengers who did not survive) has a sharp peak at lower fare values, indicating that the majority of passengers who didn't survive paid very low fares. The blue distribution (representing passengers who survived) has a broader peak that extends across a range of fares, which indicates that passengers who survived paid a mix of fares, with a significant number paying higher fares.")
else:
    st.error(f"CSV file '{csv_file_path}' not found.")







