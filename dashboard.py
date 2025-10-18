import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ------------------ Session state to manage pages ------------------
if "page" not in st.session_state:
    st.session_state.page = "main"  # Default page is main page

# ------------------ Landing Page ------------------
def main_page():
    st.markdown(
        """
        <h1 style="text-align:center; color:#E63946; font-size:47px; font-family:Arial Black;">
            ⚡ ThunderBytes 2025 ⚡
        </h1>
        <p style="text-align:center; font-size:22px; color:#333;">
            Welcome to the Heart Disease Dashboard<br>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Next button
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #E63946;
            color: white;
            font-size: 22px;
            padding: 12px 28px;
            border-radius: 8px;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Use columns to center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡ Next"):
            st.session_state.page = "dashboard"
            st.rerun()

# ------------------ Dashboard Page ------------------
def dashboard_page():
    # --- Dashboard Title (Centered and a little down) ---
    st.markdown(
        """
        <div style="text-align:center; padding-top:50px; padding-bottom:30px;">
            <h1 style="color:#ccff33; font-size:30px; font-family:Arial Black;">
                ❤️ Heart Health Analytics Dashboard ❤️
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

    # --- Load Dataset ---
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        st.error("Error: The 'heart.csv' file was not found. Please upload the file.")
        return

    # --- Sidebar filters ---
    st.sidebar.header("🔎 Choose Filters:")
    sex = st.sidebar.multiselect("Select Sex", df["sex"].unique())
    cp = st.sidebar.multiselect("Select Chest Pain Type", df["chest_pain_type"].unique())
    exercise = st.sidebar.multiselect("Exercise Angina?", df["exercise_angina"].unique())
    target = st.sidebar.multiselect("Heart Disease Status", df["target"].unique())

    # Back button at the bottom of sidebar
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)  # spacing
    if st.sidebar.button("⬅ Back"):
        st.session_state.page = "main"
        st.rerun()
    if st.sidebar.button("Predictor Page ➡"):
        st.session_state.page = "predictor"
        st.rerun()

    # Apply filters
    filtered_df = df.copy()
    if sex:
        filtered_df = filtered_df[filtered_df["sex"].isin(sex)]
    if cp:
        filtered_df = filtered_df[filtered_df["chest_pain_type"].isin(cp)]
    if exercise:
        filtered_df = filtered_df[filtered_df["exercise_angina"].isin(exercise)]
    if target:
        filtered_df = filtered_df[filtered_df["target"].isin(target)]
    
    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your selections.")
        return

    st.subheader(f"📊 Showing {filtered_df.shape[0]} filtered records")
    st.dataframe(filtered_df)

    # --- Charts Section ---
    
    # Sex-wise Heart Disease Cases
    st.markdown("<h3 style='text-align: center;'>Sex-wise Heart Disease Cases</h3>", unsafe_allow_html=True)
    fig1 = px.histogram(filtered_df, x="sex", color="target", barmode="group", text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chest Pain Type Distribution
    st.markdown("<h3 style='text-align: center;'>Chest Pain Type Distribution</h3>", unsafe_allow_html=True)
    chest_pain_counts = filtered_df['chest_pain_type'].value_counts().reset_index()
    chest_pain_counts.columns = ['chest_pain_type', 'count']
    fig2 = px.pie(
        chest_pain_counts, 
        values='count', 
        names='chest_pain_type', 
        title="Chest Pain Type", 
        hole=0
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Age Distribution
    st.markdown("<h3 style='text-align: center;'>📈 Age Distribution</h3>", unsafe_allow_html=True)
    bins = [0, 40, 60, 77]  # Defines the boundaries for the categories
    labels = ['Young Adult (≤40)', 'Middle-Aged (41-60)', 'Senior (61+)']
    filtered_df['age_category'] = pd.cut(filtered_df['age'], bins=bins, labels=labels, right=True)     
    age_category_counts = filtered_df['age_category'].value_counts().reset_index()
    age_category_counts.columns = ['age_category', 'count']
    fig = px.bar(
            age_category_counts, 
            x='age_category', 
            y='count',
            title="Age Distribution by Category",
            text_auto=True 
        )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot: Cholesterol vs Max Heart Rate
    st.markdown("<h3 style='text-align: center;'>💓 Cholesterol vs Max Heart Rate</h3>", unsafe_allow_html=True)
    fig_bubble = px.scatter(
        filtered_df,
        x="cholesterol",
        y="max_heart_rate",
        size="age",  # Bubble size based on age
        color="target",  # Color by target (e.g., heart disease presence)
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.7,
        title="Bubble Chart: Cholesterol vs Max Heart Rate"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    st.markdown("<h3 style='text-align: center;'>Heart Disease by Sex, Chest Pain Type & Target</h3>", unsafe_allow_html=True)
    fig = px.bar(filtered_df, 
            x='sex', 
            y='age', 
            color='chest_pain_type', 
            barmode='group', 
            facet_col='target', 
            title='Heart Disease by Sex, Chest Pain Type & Target')
    st.plotly_chart(fig, use_container_width=True)

    # KDE Plot for Age and Cholesterol - New Section
    st.markdown("<h3 style='text-align: center;'>Density Curves for Age and Cholesterol</h3>", unsafe_allow_html=True)
    # Use Matplotlib for the KDE plot
    plt.style.use('ggplot')
    fig_kde, ax = plt.subplots(figsize=(10, 6))
    
    # Create the KDE plot for 'age' and 'cholesterol' using Seaborn
    sns.kdeplot(data=filtered_df, x='age', label='Age', color='blue', fill=True, ax=ax)
    sns.kdeplot(data=filtered_df, x='cholesterol', label='Cholesterol', color='red', fill=True, ax=ax)
    
    ax.set_title('KDE Plot of Age and Cholesterol')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig_kde)
    
    # --- Combined Grouped Bar Chart ---
    st.markdown("<h3 style='text-align: center;'>Heart Disease by Sex and Fasting Blood Sugar</h3>", unsafe_allow_html=True)
    fig_combined = px.bar(
        filtered_df,
        x='sex',
        y='age',
        color='fasting_blood_sugar',
        barmode='group',
        facet_col='target',
        title='Proportion of Heart Disease by Sex and Fasting Blood Sugar',
        labels={'target': 'Heart Disease Status'}
    )
    st.plotly_chart(fig_combined, use_container_width=True)

    # --- Heatmap (Correlation) ---
    st.markdown("<h3 style='text-align: center;'>Correlation Heatmap</h3>", unsafe_allow_html=True)
    numeric_cols = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate', 'st_depression', 'major_vessels']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heatmap)
    ax_heatmap.set_title('Correlation Matrix of Numeric Features')
    st.pyplot(fig_heatmap)

    # --- Radar Chart ---
    st.markdown("<h3 style='text-align: center;'>Average Health Metrics by Target</h3>", unsafe_allow_html=True)
    
    # Select columns for the radar chart
    radar_cols = ['blood_pressure', 'cholesterol', 'max_heart_rate', 'st_depression']
    
    # Normalize the data for a meaningful comparison
    scaler = MinMaxScaler()
    df_normalized = filtered_df.copy()
    df_normalized[radar_cols] = scaler.fit_transform(df_normalized[radar_cols])
    
    # Group by target and calculate the mean
    radar_df = df_normalized.groupby('target')[radar_cols].mean().reset_index()
    radar_df = radar_df.melt(id_vars='target', var_name='Metric', value_name='Normalized Average')

    # Create the radar chart
    fig_radar = px.line_polar(
        radar_df, 
        r='Normalized Average', 
        theta='Metric', 
        color='target', 
        line_close=True,
        title='Average Normalized Metrics across Target Groups'
    )
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Filtered Data", data=csv, file_name="filtered_heart_data.csv", mime="text/csv")

    # --- Show Pivot Table ---
    st.markdown("<h3 style='text-align: center;'>📊 Heart Disease Summary by Sex & Chest Pain Type</h3>", unsafe_allow_html=True)
    pivot_df = pd.pivot_table(filtered_df, values="age", index=["sex"], columns="chest_pain_type", aggfunc="count")
    st.write(pivot_df.style.background_gradient(cmap="Reds"))

# ------------------ Prediction Page ------------------
def predictor_page():
    st.markdown(
        """
        <div style="text-align:center; padding-top:50px; padding-bottom:30px;">
            <h1 style="color:#20B2AA; font-size:30px; font-family:Arial Black;">
                🤔 Heart Disease Predictor 🤔
            </h1>
            <p>Enter a patient's details to predict their heart disease status.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    # --- Load Data and Train Model ---
    try:
        data = pd.read_csv("heart.csv")
    except FileNotFoundError:
        st.error("Error: The 'heart.csv' file was not found.")
        return
        
    data_filtered = data[data['age'] >= 60].copy()
    if data_filtered.empty:
        st.warning("The dataset does not contain enough data for individuals aged 60+ to train the model.")
        return

    data_processed = pd.get_dummies(data_filtered, columns=['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_angina', 'st_slope', 'blood disorder', 'target'], drop_first=True)
    X = data_processed.drop('target_no heart disease', axis=1)
    y = data_processed['target_no heart disease']

    if y.nunique() < 2:
        st.warning("The filtered dataset contains only one class for the target variable.")
        return

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    st.success("Prediction model is ready!")

    # --- User Input Form ---
    st.subheader("Enter Patient Details ")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=60, max_value=120, value=65)
        sex = st.selectbox("Sex", data['sex'].unique())
        cp = st.selectbox("Chest Pain Type", data['chest_pain_type'].unique())
        bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar", data['fasting_blood_sugar'].unique())
        ecg = st.selectbox("Resting Electrocardiographic Results", data['rest_ecg'].unique())
        hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        ex_angina = st.selectbox("Exercise Induced Angina", data['exercise_angina'].unique())
        st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0)
        st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", data['st_slope'].unique())
        major_vessels = st.slider("Number of Major Vessels", 0, 4, 1)
        blood_disorder = st.selectbox("Blood Disorder", data['blood disorder'].unique())

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create a new data point for prediction
        input_data = {
            'age': age,
            'sex': sex,
            'chest_pain_type': cp,
            'blood_pressure': bp,
            'cholesterol': cholesterol,
            'fasting_blood_sugar': fbs,
            'rest_ecg': ecg,
            'max_heart_rate': hr,
            'exercise_angina': ex_angina,
            'st_depression': st_depression,
            'st_slope': st_slope,
            'major_vessels': major_vessels,
            'blood disorder': blood_disorder
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # One-hot encode the new data to match the training data format
        processed_input = pd.get_dummies(input_df, columns=['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_angina', 'st_slope', 'blood disorder'])

        # Reindex the input to match the training data columns, filling missing with 0
        final_input = processed_input.reindex(columns=X.columns, fill_value=0)

        # Make prediction
        prediction = model.predict(final_input)
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.markdown(
                """
                <h2 style='color:red;'>
                    💔 Based on the data, the model predicts that this person has heart disease.
                </h2>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <h2 style='color:green;'>
                    ❤️ Based on the data, the model predicts that this person does not have heart disease.
                </h2>
                """, unsafe_allow_html=True)


# ------------------ Page Navigation ------------------
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "predictor":
    predictor_page()