import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB as NaiveBayesClassifier
from sklearn.metrics import accuracy_score,f1_score
import joblib

Tree_model = DecisionTreeClassifier()
Bayesian_model = NaiveBayesClassifier()

feature_cols = ['N', 'P', 'K','ph', 'rainfall']
target_col = 'label'

#do some basic visualizations
# Visualize the distribution of the target variable
#visualize_data(df, feature_cols)

def load_data():
	try:
		df = pd.read_csv("C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\Crop_recommendation.csv")

		print("Dimensions of loaded data {}".format(df.shape))
	except FileNotFoundError:
		print("File not found: C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\Crop_recommendation.csv")
	return df

def exploratory_data_analysis(df):
    print("DataFrame Info:")
    print(df.info())

    #print("\nSample Data:")
    #print(df.sample(5))

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    print("\nUnique Values per Column:")
    print(df['label'].value_counts())


def preprocess_data(df):
	# Remove rows with missing values
	df.dropna(inplace=True)
	duplicates = df[df.duplicated()]
	#print(f"Number of duplicate rows: {len(duplicates)}")
	if not duplicates.empty:
		print("Duplicate rows found:")
		print(duplicates)

	# One-hot encoding for the target variable (label)
	ohe = OneHotEncoder(sparse_output=False)
	Y_train_encoded = ohe.fit_transform(Y_train.reshape(-1, 1))
	Y_test_encoded = ohe.transform(Y_test.reshape(-1, 1))

	X_train_scaled,X_test_scaled=feature_scaling(X_train,X_test)

	return X_train_scaled, X_test_scaled, Y_train_encoded, Y_test_encoded
	
def visualize_data(df, feature_cols):

	plt.figure(figsize=(10, 6))
	df['label'].value_counts().plot(kind='pie', title='Crop Distribution')
	#save the figure 
	plt.savefig('C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\plots\\crop_distribution.png')

	for feature in feature_cols:
		plt.figure(figsize=(8, 5))
		sns.kdeplot(df[feature], fill=True, color='skyblue', linewidth=2)
		plt.title(f'Distribution of {feature}')
		plt.xlabel(feature)
		plt.ylabel('Density')
		plt.savefig(f'C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\plots\\{feature}_distribution.png')
		plt.close()
	#plt.show()  # Uncomment to display the plot

	#generate box plots for each feature
	for feature in feature_cols:
		plt.figure(figsize=(8, 5))
		sns.boxplot(x=df[feature])
		plt.title(f'Box Plot of {feature}')
		plt.xlabel(feature)
		plt.savefig(f'C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\plots\\{feature}_boxplot.png')
		plt.close()

	if 'label' in df.columns:
		df = pd.get_dummies(df, columns=['label'], drop_first=True)
	return df



def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled



def train_Treemodel(x_train_scaled, y_train_encoded):
	# Placeholder for model training logic
	#model = DecisionTreeClassifier()
	Tree_model.fit(x_train_scaled, y_train_encoded)
	#Bayesian_model.fit(x_train, y_train)
	print("Model trained successfully.")
	# Visualize the decision tree
	plt.figure(figsize=(20, 10))
	plot_tree(Tree_model, filled=True, feature_names=['N', 'P', 'K',  'ph', 'rainfall'])
	plt.savefig('C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\plots\\decision_tree.png')
	#print("Decision tree visualized and saved.")

def train_Bayesianmodel(x_train,y_train):
	# Placeholder for model training logic
	Bayesian_model.fit(x_train, y_train)
	#print("Naive Bayes model trained successfully.")


def predictions(x_test_scaled):
	# Placeholder for making predictions
	print("Making predictions on the test set...")
	y_pred = Tree_model.predict(x_test_scaled)
	return y_pred


def predictions_naive_bayes(x_test):
	# Placeholder for making predictions with Naive Bayes
	#print("Making predictions on the test set using Naive Bayes...")
	y_pred = Bayesian_model.predict(x_test)
	return y_pred

def evaluate_model(y_test, y_pred):
	accuracy = accuracy_score(y_test, y_pred)
	#print(f"Accuracy: {accuracy:.2f}")
	f1 = f1_score(y_test, y_pred, average='weighted')
	#print(f"F1 Score: {f1:.2f}")

	return accuracy, f1

#function to evaluate model with random data
def evaluate_with_random_data(model, feature_cols,df):
	# Generate random data within the range of each feature
	random_data = {}
	for col in feature_cols:
		min_val = df[col].min()
		max_val = df[col].max()
		random_data[col] = np.random.uniform(min_val, max_val, 10)  # Generate 10 random samples

	random_df = pd.DataFrame(random_data)
	
	# Scale the random data
	scaler = StandardScaler()
	random_data_scaled = scaler.fit_transform(random_df)

	# Make predictions
	predictions = model.predict(random_data_scaled)
	print("Predictions on random data:")
	#print(predictions)
	return predictions,None

'''def generate_crop_recommendation_data(n_samples=100, as_numpy=False, as_dict=False):
    # Crop nutrient & environmental ranges
    crop_data = {
        "Rice": {"N": (90, 120), "P": (40, 60), "K": (40, 60), "pH": (5.5, 6.5), "Rainfall": (1200, 2000)},
        "Maize": {"N": (120, 180), "P": (50, 80), "K": (40, 60), "pH": (5.5, 7.5), "Rainfall": (500, 800)},
        "Potato": {"N": (100, 150), "P": (60, 80), "K": (100, 150), "pH": (5.0, 6.5), "Rainfall": (600, 800)},
        "Groundnut": {"N": (10, 30), "P": (40, 60), "K": (40, 60), "pH": (6.0, 7.5), "Rainfall": (500, 1000)},
        "Tomato": {"N": (50, 100), "P": (40, 60), "K": (60, 100), "pH": (6.0, 7.0), "Rainfall": (600, 1200)},
    }

    rows = []
    np.random.seed(42)  # reproducibility

    for _ in range(n_samples):
        crop = np.random.choice(list(crop_data.keys()))
        vals = crop_data[crop]
        N = np.random.uniform(*vals["N"])
        P = np.random.uniform(*vals["P"])
        K = np.random.uniform(*vals["K"])
        pH = np.random.uniform(*vals["pH"])
        rainfall = np.random.uniform(*vals["Rainfall"])
        rows.append([N, P, K, pH, rainfall, crop])

    df = pd.DataFrame(rows, columns=["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Crop"])

    if as_numpy:
        return df.drop(columns=["Crop"]).to_numpy()
    elif as_dict:
        return df.to_dict(orient="list")
    else:
        return df'''


#load the data
df = load_data()

X = df[feature_cols].values
Y = df[target_col].values

#agile-approach to model training and predictions using iterative training and evaluation
n_iterations=1
tree_accuracies=[]
bayes_accuracies=[]
tree_f1s=[]
bayes_f1s=[]

for i in range(n_iterations):
	# Split train/test data
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=i, shuffle=True)

	#perform exploratory data analysis
	exploratory_data_analysis(df)

	#preprocess the data
	x_train_scaled, x_test_scaled, y_train_encoded, y_test_encoded = preprocess_data(df)

	#train models
	train_Treemodel(x_train_scaled, y_train_encoded)
	train_Bayesianmodel(X_train, Y_train)

	#perform predictions
	y_pred1 = predictions(x_test_scaled)
	y_pred2 = predictions_naive_bayes(X_test)

	# Evaluate Decision Tree Classifier
	tree_acc, tree_f1score = evaluate_model(y_test_encoded, y_pred1)
	#print(f"Decision Tree Accuracy: {tree_acc:.2f}, F1 Score: {tree_f1score:.2f}")

	#Evaluate Naive Bayes classifier
	bayes_acc, bayes_f1score = evaluate_model(Y_test, y_pred2)
	bayes_accuracies.append(bayes_acc)
	bayes_f1s.append(bayes_f1score)

	#print(f"Naive Bayes Accuracy: {bayes_acc:.2f}, F1 Score: {bayes_f1score:.2f}")


	#print("Random Data Evaluation:")
	#print(f"Iteration {i+1}: Decision Tree Predictions={random_tr_accuracies} | Naive Bayes Predictions={random_bay_accuracies}")


print(f"\nAverage Decision Tree Accuracy over {n_iterations} iterations: {np.mean(tree_accuracies):.3f} ± {np.std(tree_accuracies):.3f}")
print(f"Average Decision Tree F1 Score over {n_iterations} iterations: {np.mean(tree_f1s):.3f} ± {np.std(tree_f1s):.3f}")
print(f"Average Naive Bayes Accuracy over {n_iterations} iterations: {np.mean(bayes_accuracies):.3f} ± {np.std(bayes_accuracies):.3f}")
print(f"Average Naive Bayes F1 Score over {n_iterations} iterations: {np.mean(bayes_f1s):.3f} ± {np.std(bayes_f1s):.3f}")



import os
os.makedirs('C:\\Users\\DELL\\ML_Int\\models', exist_ok=True)

#save the models
joblib.dump(Tree_model, 'C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\models\\Tree_model.pkl')
joblib.dump(Bayesian_model, 'C:\\Users\\DELL\\SmartFarmingApp\\CropRecommendationSystem\\ML_Int\\models\\Bayesian_model.pkl')
print("Models dumped")