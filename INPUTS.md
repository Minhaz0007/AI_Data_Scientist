File: advanced_analysis.py
  - multiselect: Select Columns for PCA
  - multiselect: Select Columns for t-SNE
  - multiselect: Select Columns for Anomaly Detection
  - selectbox: Select Text Column
  - slider: Number of Components
  - selectbox: Color Points By
  - slider: Perplexity
  - slider: Iterations
  - selectbox: Color Points By
  - selectbox: Detection Method
  - slider: Expected Anomaly Proportion

File: analysis.py
  - multiselect: Select Features for Clustering
  - slider: Number of Clusters
  - selectbox: Select Test
  - selectbox: Select Column
  - selectbox: Detection Method
  - selectbox: Select Column
  - slider: IQR Multiplier
  - slider: Z-Score Threshold
  - selectbox: Column 1
  - selectbox: Column 2
  - selectbox: Treatment Method
  - selectbox: Column 1
  - selectbox: Column 2
  - checkbox: Append cluster labels to current dataset
  - selectbox: Column 1
  - selectbox: Column 2

File: chat.py
  - radio: Provider
  - text_input: API Key (Optional if set in Env)
  - selectbox: Model

File: cleaning.py
  - selectbox: Select Column
  - selectbox: Imputation Strategy
  - text_input: Enter Constant Value

File: dashboard.py
  - selectbox: Layout

File: feature_engineering.py
  - multiselect: Select Columns for Polynomial Features
  - slider: Polynomial Degree
  - multiselect: Select Columns for Interactions
  - selectbox: Select Column to Convert
  - selectbox: Select DateTime Column
  - selectbox: Select Column to Bin
  - multiselect: Select Feature Columns (X)
  - selectbox: Select Target Column (y)
  - selectbox: Selection Method
  - slider: Number of Features to Select
  - slider: Number of Bins
  - selectbox: Binning Strategy

File: ingestion.py
  - file_uploader: Upload your dataset (drag and drop supported)
  - text_input: Connection String (e.g., sqlite:///data.db)
  - text_area: SQL Query
  - checkbox: Show Data Info

File: insights.py
  - radio: Select AI Provider
  - text_input: Enter API Key (Optional)
  - selectbox: Select Model
  - slider: Max Response Tokens

File: modeling.py
  - radio: Select Task Type
  - multiselect: Select Feature Columns (X)
  - selectbox: Select Target Column (y)
  - selectbox: Select Model
  - slider: Test Set Size
  - checkbox: Run AutoML (Compare All Models)

File: profiling.py
  - selectbox: Select Column to visualize

File: reporting.py
  - text_input: Report Title
  - radio: Select Report Format

File: timeseries.py
  - selectbox: Data Frequency
  - selectbox: Select Date/Time Column
  - selectbox: Select Value Column
  - slider: Forecast Periods
  - selectbox: Forecasting Method

File: transformation.py
  - selectbox: Select Column to Filter
  - selectbox: Condition
  - multiselect: Group By Column(s)
  - selectbox: Aggregate Column
  - selectbox: Aggregation Method
  - radio: Operation
  - text_input: New Column Name
  - text_area: Expression
  - file_uploader: Upload second dataset
  - text_input: Value
  - multiselect: ID Columns (keep fixed)
  - multiselect: Value Columns (to unpivot)
  - text_input: Variable Column Name
  - text_input: Value Column Name
  - text_input: Min Value
  - text_input: Max Value
  - selectbox: Row Index
  - selectbox: Column Headers
  - selectbox: Aggregation
  - selectbox: Merge Type
  - selectbox: Values
  - multiselect: Merge On (common columns)
  - selectbox: Left Dataset Column
  - selectbox: Right Dataset Column

File: visualization.py
  - selectbox: Select Chart Type
  - selectbox: Color (Optional)
  - selectbox: Size (Optional)
  - selectbox: X Axis
  - selectbox: Y Axis
  - selectbox: Color (Optional)
  - selectbox: X Axis (Time/Sequence)
  - selectbox: Y Axis
  - selectbox: Color (Optional)
  - selectbox: Bar Mode
  - selectbox: X Axis (Categorical)
  - selectbox: Y Axis (Numerical)
  - selectbox: Column
  - slider: Number of Bins
  - selectbox: Color (Optional)
  - selectbox: Color (Optional)
  - selectbox: Numerical Column
  - selectbox: Categorical Column (Optional)
  - multiselect: Select Columns
  - selectbox: Labels (Categorical)
  - selectbox: Values (Numerical)
  - multiselect: Hierarchy Path (Select in order)
  - selectbox: Values
  - multiselect: Hierarchy Path
  - selectbox: Values
  - selectbox: Values
  - selectbox: Stages
  - selectbox: Color (Group)
  - selectbox: Radius (Numerical)
  - selectbox: Angle (Categorical)
  - selectbox: Color (Stack)
  - selectbox: X Axis
  - selectbox: Y Axis
  - selectbox: Color (Optional)
  - selectbox: Numerical Data
  - selectbox: Category (Optional)
